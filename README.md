# RAG Q&A API

[![Docker](https://github.com/lucianhal9000/rag-qa-api/actions/workflows/docker.yml/badge.svg)](https://github.com/lucianhal9000/rag-qa-api/actions/workflows/docker.yml)

A Retrieval-Augmented Generation (RAG) API built with **FastAPI**,
**LangChain**, **FAISS**, and **Groq (LLaMA 3)**, containerized and covered by
a 29-test suite that runs in CI on every push.

See [Scope and limitations](#scope-and-limitations) before treating this as
something to run in production — it is a portfolio project, and the gaps are
documented rather than glossed over.

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

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

29 tests, roughly half a second, no API key and no network required. The suite
injects a deterministic hashing embedding and a stub LLM in place of
sentence-transformers and Groq; FAISS, the text splitter, the LCEL chain, and
FastAPI routing are all exercised for real.

The fake embedding hashes tokens into fixed buckets rather than returning random
vectors, so documents sharing words genuinely land near each other. That makes
retrieval assertions meaningful — the suite can check that a question about
Redis returns the Redis chunk, not just that *some* chunk came back.

### Defects the suite caught

| Defect | Symptom before the fix |
|--------|------------------------|
| Empty upload reached FAISS with zero documents | `IndexError` surfacing to the caller as a 500 instead of a 400 |
| `os.unlink` ran only on the success path | Failed ingest leaked its temp file onto disk on every attempt |
| No upload size limit | Whole file read into memory regardless of size; now capped at 10 MB with a 413 |

## Scope and limitations

Known and deliberate, rather than discovered later:

- **Single shared index.** The pipeline is one process-wide instance, so every
  caller shares one vector store. Fine for a demo; a real deployment needs
  per-tenant namespacing.
- **No persistence.** The FAISS index is in memory, so ingested documents are
  lost on restart. `FAISS.save_local()` / `load_local()` onto a mounted volume
  is the fix.
- **No authentication**, and CORS is open to all origins.
- **Dependencies are unpinned**, so builds are reproducible only until an
  upstream release changes behaviour.

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
