# RAG Q&A API

[![Docker](https://github.com/lucianhal9000/rag-qa-api/actions/workflows/docker.yml/badge.svg)](https://github.com/lucianhal9000/rag-qa-api/actions/workflows/docker.yml)

A production-ready Retrieval-Augmented Generation (RAG) API built with
**FastAPI**, **LangChain**, **FAISS**, and **Groq (LLaMA 3)**.

## Architecture

```
User → FastAPI → LangChain RAG Chain → FAISS Vector Store
                                     → Groq LLM (LLaMA 3)
```

## Endpoints

| Method | Route          | Description               |
|--------|----------------|---------------------------|
| GET    | `/health`      | Health check              |
| POST   | `/ingest/text` | Ingest raw text           |
| POST   | `/ingest/file` | Upload .txt or .pdf       |
| POST   | `/query`       | Ask a question            |

## Stack

- **FastAPI** — async REST backend
- **LangChain LCEL** — RAG chain with prompt, retriever, and LLM
- **FAISS** — local vector similarity search
- **HuggingFace Embeddings** — `all-MiniLM-L6-v2` (runs locally, free)
- **Groq API** — free LLaMA 3 inference

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/lucianhal9000/rag-qa-api.git
cd rag-qa-api

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
cp .env.example .env
# Edit .env and paste your key from https://console.groq.com

# 5. Run the server
uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API explorer.

## Run with Docker

```bash
cp .env.example .env      # then paste your Groq key into .env
docker compose up --build
```

The API is then on `http://localhost:8000`, with `/docs` and `/health` available
as usual. To run the image without Compose:

```bash
docker build -t rag-qa-api .
docker run -p 8000:8000 --env-file .env rag-qa-api
```

### Image notes

- The `all-MiniLM-L6-v2` embedding model is downloaded **at build time**, not at
  container start. Without this the first request after every cold start waits on
  a ~90MB HuggingFace download, and the container cannot start at all on a host
  with no outbound internet.
- `torch` is installed from the CPU-only wheel index. The default PyPI wheel
  bundles CUDA libraries that are dead weight here and add roughly 2.5GB.
- A `HEALTHCHECK` polls `/health` every 30s, with a 40s start period covering
  model load into memory. `docker ps` reports the container as `healthy` once the
  API is actually serving.
- The container runs as a non-root `appuser`.

### Known limitation

The FAISS index lives in memory only, so ingested documents are lost when the
container restarts. Persisting it via `FAISS.save_local()` / `load_local()` onto a
mounted volume is the next step.

## Example Usage

### Ingest text
```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "LangChain is a framework for building LLM applications."}'
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LangChain used for?"}'
```
