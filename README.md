# Hukuk RAG

## Yerel gelistirme

AI agent ile `rag.py` dosyasini VS Code'da gelistirin. `requirements.txt` Colab ortaminda kurulur. `.env` dosyasini Git'e eklemeyin.

## Colab kullanimi

1. Bu klasoru GitHub repository olarak yayinlayin.
2. `hukuk_rag.ipynb` icindeki `REPO_URL` satirina repository adresini yazin.
3. Google Drive'da `My Drive/hukuk_rag/pdf/` klasorune PDF'leri koyun.
4. `.env` dosyasini `My Drive/hukuk_rag/.env` konumuna koyun veya Colab Secrets icinde `HF_TOKEN` tanimlayin.
5. Colab'da T4 GPU secip notebook hucrelerini sirayla calistirin.

Kod her calistirmada GitHub'dan guncel `rag.py` dosyasini alir. PDF'ler ve FAISS indeksi Drive'da kalir. Llama agirliklari `/content/huggingface` altinda Colab cache'inde tutulur.
