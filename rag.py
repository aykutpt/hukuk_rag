import json
import os
import pickle
import re
from pathlib import Path

import faiss
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class HukukRAG:
    def __init__(
        self,
        pdf_dir,
        vector_db_dir="vector_db",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        cache_dir=None,
    ):
        self.pdf_dir = Path(pdf_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.embedding_model_name = embedding_model
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.embedder = None
        self.index = None
        self.records = []
        self.tokenizer = None
        self.model = None

    @staticmethod
    def normalize_text(text):
        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def split_text(cls, text, chunk_size=900, overlap=150):
        words = cls.normalize_text(text).split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            if chunk := " ".join(words[start:end]):
                chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap
        return chunks

    def extract_records(self):
        pdf_paths = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            raise FileNotFoundError(f"PDF bulunamadi: {self.pdf_dir}")

        records = []
        for pdf_path in pdf_paths:
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = self.normalize_text(page.extract_text())
                for chunk_number, text in enumerate(self.split_text(page_text), start=1):
                    records.append(
                        {
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk": chunk_number,
                        }
                    )
        if not records:
            raise ValueError("PDF metni cikarilamadi. Taranmis PDF'ler icin OCR gerekir.")
        return records

    def build_index(self, batch_size=32):
        self.embedder = SentenceTransformer(
            self.embedding_model_name,
            cache_folder=self.cache_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.records = self.extract_records()
        texts = [record["text"] for record in self.records]
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.save_index()
        return len(self.records)

    def save_index(self):
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        with (self.vector_db_dir / "metadata.pkl").open("wb") as file:
            pickle.dump(self.records, file)
        with (self.vector_db_dir / "chunks.json").open("w", encoding="utf-8") as file:
            json.dump(self.records, file, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(self.vector_db_dir / "hukuk.index"))

    def load_index(self):
        if not (self.vector_db_dir / "hukuk.index").exists():
            return self.build_index()
        self.embedder = SentenceTransformer(
            self.embedding_model_name,
            cache_folder=self.cache_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.index = faiss.read_index(str(self.vector_db_dir / "hukuk.index"))
        with (self.vector_db_dir / "metadata.pkl").open("rb") as file:
            self.records = pickle.load(file)
        if self.index.ntotal != len(self.records):
            raise ValueError("FAISS indeksi ile metadata kayit sayisi uyusmuyor.")

    def load_llama(self, token):
        if not token or not token.strip():
            raise ValueError("Hugging Face tokeni gerekli.")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=self.cache_dir, token=token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            cache_dir=self.cache_dir,
            token=token,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def retrieve(self, question, top_k=5):
        if self.index is None or self.embedder is None:
            raise RuntimeError("Once load_index() veya build_index() cagrilmalidir.")
        query = self.embedder.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, positions = self.index.search(query, top_k)
        results = []
        for score, position in zip(scores[0], positions[0]):
            if position >= 0:
                record = self.records[position].copy()
                record["score"] = float(score)
                results.append(record)
        return results

    def ask(self, question, top_k=5, max_new_tokens=500):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Once load_llama() cagrilmalidir.")
        results = self.retrieve(question, top_k)
        context = "\n\n".join(
            f"[KAYNAK {number}] {item['source']} - sayfa {item['page']}\n{item['text']}"
            for number, item in enumerate(results, start=1)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Yalnizca KAYNAKLAR bolumundeki bilgilere dayanarak Turkce cevap ver. "
                    "Cevap kaynaklarda yoksa bunu acikca soyle. Hukuki tavsiye verme; "
                    "genel bilgi oldugunu belirt. Cevabin sonunda [KAYNAK N] kullan."
                ),
            },
            {"role": "user", "content": f"KAYNAKLAR:\n{context}\n\nSORU: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        return answer, results