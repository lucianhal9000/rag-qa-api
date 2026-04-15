# RAG Q&A API

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
