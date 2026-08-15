# RAG Q&A API

[![Docker](https://github.com/lucianhal9000/rag-qa-api/actions/workflows/docker.yml/badge.svg)](https://github.com/lucianhal9000/rag-qa-api/actions/workflows/docker.yml)

A Retrieval-Augmented Generation (RAG) API built with **FastAPI**,
**LangChain**, **FAISS**, and **Groq (LLaMA 3)**, containerized and covered by
a 31-test suite that runs in CI on every push.

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

31 tests, a few seconds, no API key and no network required. The suite
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
| No upload size limit | Whole file read into memory regardless of size |
| Size cap enforced after reading the body | A 200 MB upload was fully materialized before the 413; uploads now stream to disk in 64 KB chunks and abort one chunk past the 10 MB cap |

### Defects only running the container caught

The test suite is green without a network or an API key, and CI builds the image
on every push — but neither exercises a *cold start of the real image*. Running
it locally surfaced a defect that no green build could have shown:

The embedding model is baked in at build time, but `sentence-transformers` still
issued around forty `HEAD` requests to `huggingface.co` on every boot to
revalidate that cache, complete with an unauthenticated-rate-limit warning. The
weights loaded instantly from disk; the container simply refused to start
without internet anyway. `HF_HUB_OFFLINE=1` closed the gap — see
[Image notes](#image-notes).

## Scope and limitations

Known and deliberate, rather than discovered later:

- **Single shared index.** The pipeline is one process-wide instance, so every
  caller shares one vector store. Fine for a demo; a real deployment needs
  per-tenant namespacing.
- **No persistence.** The FAISS index is in memory, so ingested documents are
  lost on restart. `FAISS.save_local()` / `load_local()` onto a mounted volume
  is the fix.
- **`faiss-cpu` falls back to the generic kernel.** The wheel ships without
  `swigfaiss_avx2`, so similarity search runs unvectorized. Immaterial at demo
  corpus sizes, measurable at scale.
- **No authentication**, and CORS is open to all origins.
- **`GROQ_API_KEY` is required at startup, not on first request.** Without it
  the app raises during FastAPI's `lifespan` and the container exits with code
  3. This is deliberate fail-fast: a process that boots and then fails every
  query is harder to diagnose than one that refuses to start.
- **`/health` is a liveness probe, not a readiness probe.** It reports `ok`
  even with an empty index. `vectorstore_ready` and `indexed_vectors` are
  exposed in the response body so a caller can distinguish the two, but the
  status does not gate on them — an empty index is a valid state, not a
  failure.
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
  container start, and `HF_HUB_OFFLINE=1` stops sentence-transformers
  revalidating it against the Hub on every boot. The image is therefore
  genuinely self-contained — it starts with no outbound internet at all:

  ```bash
  docker run --rm --network none --env-file .env rag-qa-api
  ```

  Queries still need network for the Groq call; only the embedding path is
  offline.
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