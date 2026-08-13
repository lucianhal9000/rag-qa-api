import os
import logging
import tempfile
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline, EmptyDocumentError

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
CHUNK_BYTES = 64 * 1024
ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@lru_cache(maxsize=1)
def get_rag() -> RAGPipeline:
    """Process-wide pipeline, built on first use.

    Exposed as a FastAPI dependency so tests can override it with a pipeline
    backed by fake embeddings and a fake LLM.
    """
    return RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG Q&A API starting up...")
    # Load the model at startup rather than on the first request, so a broken
    # config fails immediately instead of at the first user's expense. Skipped
    # when a test has overridden the dependency.
    if get_rag not in app.dependency_overrides:
        get_rag()
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

async def _spool_upload(file: UploadFile, ext: str) -> str:
    """Stream an upload to a temp file, aborting as soon as it exceeds the cap.

    Reading the body in one call would materialize the whole upload in memory
    before the size check could run, so an oversized request could exhaust a
    worker before being rejected. Streaming holds one chunk at a time and stops
    at the first byte past the limit.

    The file is closed before unlinking because Windows refuses to delete an
    open handle.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp_path = tmp.name
    size = 0
    try:
        while chunk := await file.read(CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds the 10 MB limit.")
            tmp.write(chunk)
    except BaseException:
        tmp.close()
        os.unlink(tmp_path)
        raise
    tmp.close()

    if size == 0:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return tmp_path


@app.get("/health")
async def health(rag: RAGPipeline = Depends(get_rag)):
    """Liveness and readiness check. Used by the container HEALTHCHECK."""
    return {
        "status": "ok",
        "version": app.version,
        "vectorstore_ready": rag.vectorstore is not None,
        "indexed_vectors": rag.indexed_vectors,
    }


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: TextIngestRequest, rag: RAGPipeline = Depends(get_rag)):
    """Ingest raw text into the vector store."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    try:
        chunks = rag.ingest_text(request.text)
    except EmptyDocumentError:
        raise HTTPException(status_code=400, detail="Text produced no indexable content.")
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(message="Text ingested successfully.", chunks_added=chunks)


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), rag: RAGPipeline = Depends(get_rag)):
    """Upload and ingest a .txt or .pdf file into the vector store."""
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    tmp_path = await _spool_upload(file, ext)
    try:
        chunks = rag.ingest_file(tmp_path)
    except EmptyDocumentError:
        raise HTTPException(status_code=400, detail="No extractable text found in the file.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # finally, not a trailing unlink: an ingest failure used to leave the
        # temp file behind on disk.
        os.unlink(tmp_path)

    return IngestResponse(
        message=f"File '{file.filename}' ingested successfully.",
        chunks_added=chunks,
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag: RAGPipeline = Depends(get_rag)):
    """Query the ingested documents."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = rag.query(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return QueryResponse(answer=result["answer"], sources=result["sources"])
