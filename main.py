import os
import logging
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

rag = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG Q&A API starting up...")
    yield
    logger.info("RAG Q&A API shutting down.")


app = FastAPI(
    title="RAG Q&A API",
    description="Upload documents and query them using LangChain + FAISS + Groq.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------

class TextIngestRequest(BaseModel):
    text: str


class IngestResponse(BaseModel):
    message: str
    chunks_added: int


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


# ---------- Routes ----------

@app.get("/health")
async def health():
    """Liveness and readiness check. Used by the container HEALTHCHECK."""
    ready = rag.vectorstore is not None
    return {
        "status": "ok",
        "version": app.version,
        "vectorstore_ready": ready,
        "indexed_vectors": rag.vectorstore.index.ntotal if ready else 0,
    }


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: TextIngestRequest):
    """Ingest raw text into the vector store."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        chunks = rag.ingest_text(request.text)
        return IngestResponse(
            message="Text ingested successfully.",
            chunks_added=chunks,
        )
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Upload and ingest a .txt or .pdf file into the vector store."""
    allowed = {".txt", ".pdf"}
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        chunks = rag.ingest_file(tmp_path)
        os.unlink(tmp_path)
        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully.",
            chunks_added=chunks,
        )
    except Exception as e:
        logger.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the ingested documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag.query(request.question)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
