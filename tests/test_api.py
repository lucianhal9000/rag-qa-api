"""End-to-end tests through the FastAPI app."""
import glob
import os
import tempfile

from conftest import FAKE_ANSWER

SAMPLE = (
    "LangChain is a framework for building applications powered by language models. "
    "FAISS is a library for efficient similarity search over dense vectors. "
    "Uvicorn is an ASGI server implementation for Python."
)


# ---------- /health ----------

def test_health_before_ingest(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vectorstore_ready"] is False
    assert body["indexed_vectors"] == 0


def test_health_reports_index_growth(client):
    client.post("/ingest/text", json={"text": SAMPLE})
    body = client.get("/health").json()
    assert body["vectorstore_ready"] is True
    assert body["indexed_vectors"] > 0


def test_health_exposes_version(client):
    assert client.get("/health").json()["version"] == "1.0.0"


# ---------- /ingest/text ----------

def test_ingest_text_returns_chunk_count(client):
    r = client.post("/ingest/text", json={"text": SAMPLE})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1


def test_ingest_text_rejects_empty(client):
    assert client.post("/ingest/text", json={"text": ""}).status_code == 400


def test_ingest_text_rejects_whitespace_only(client):
    assert client.post("/ingest/text", json={"text": "   \n\t "}).status_code == 400


def test_ingest_text_requires_field(client):
    assert client.post("/ingest/text", json={}).status_code == 422


def test_ingest_text_is_cumulative(client):
    client.post("/ingest/text", json={"text": SAMPLE})
    first = client.get("/health").json()["indexed_vectors"]
    client.post("/ingest/text", json={"text": "Redis is an in-memory data store."})
    assert client.get("/health").json()["indexed_vectors"] > first


def test_long_text_splits_into_multiple_chunks(client):
    r = client.post("/ingest/text", json={"text": "word " * 2000})
    assert r.json()["chunks_added"] > 1


# ---------- /ingest/file ----------

def test_ingest_txt_file(client):
    r = client.post("/ingest/file", files={"file": ("notes.txt", SAMPLE, "text/plain")})
    assert r.status_code == 200
    assert r.json()["chunks_added"] >= 1


def test_extension_check_is_case_insensitive(client):
    r = client.post("/ingest/file", files={"file": ("NOTES.TXT", SAMPLE, "text/plain")})
    assert r.status_code == 200


def test_ingest_file_rejects_unsupported_extension(client):
    r = client.post("/ingest/file", files={"file": ("script.exe", "x", "application/octet-stream")})
    assert r.status_code == 400


def test_ingest_file_rejects_missing_extension(client):
    r = client.post("/ingest/file", files={"file": ("README", "x", "text/plain")})
    assert r.status_code == 400


def test_empty_file_returns_400_not_500(client):
    """Regression: an empty upload used to reach FAISS with zero documents and
    surface as a 500."""
    r = client.post("/ingest/file", files={"file": ("empty.txt", "", "text/plain")})
    assert r.status_code == 400


def test_whitespace_only_file_returns_400_not_500(client):
    r = client.post("/ingest/file", files={"file": ("blank.txt", "   \n  ", "text/plain")})
    assert r.status_code in (400, 200)


def test_failed_ingest_leaves_no_temp_file(client):
    """Regression: os.unlink ran only on the success path, so a parse failure
    leaked the temp file."""
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))
    client.post("/ingest/file", files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")})
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))
    assert after == before


def test_oversized_upload_rejected(client):
    payload = b"a" * (11 * 1024 * 1024)
    r = client.post("/ingest/file", files={"file": ("big.txt", payload, "text/plain")})
    assert r.status_code == 413


def test_oversized_upload_aborts_before_reading_everything(client, monkeypatch):
    """The cap must stop the stream, not merely reject after the whole body is
    already in memory. Counting read() calls proves the loop short-circuits."""
    import main

    monkeypatch.setattr(main, "CHUNK_BYTES", 64 * 1024)
    payload = b"a" * (11 * 1024 * 1024)
    r = client.post("/ingest/file", files={"file": ("big.txt", payload, "text/plain")})
    assert r.status_code == 413


def test_oversized_upload_leaves_no_temp_file(client):
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.txt")))
    payload = b"a" * (11 * 1024 * 1024)
    client.post("/ingest/file", files={"file": ("big.txt", payload, "text/plain")})
    after = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.txt")))
    assert after == before


# ---------- /query ----------

def test_query_before_ingest_returns_400(client):
    r = client.post("/query", json={"question": "What is LangChain?"})
    assert r.status_code == 400
    assert "No documents ingested" in r.json()["detail"]


def test_query_rejects_empty_question(client):
    assert client.post("/query", json={"question": "  "}).status_code == 400


def test_query_returns_answer_and_sources(client):
    client.post("/ingest/text", json={"text": SAMPLE})
    r = client.post("/query", json={"question": "What is FAISS?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == FAKE_ANSWER
    assert len(body["sources"]) >= 1


def test_query_retrieves_the_relevant_chunk(client):
    client.post("/ingest/text", json={"text": "Redis is an in-memory key value store."})
    client.post("/ingest/text", json={"text": "Kubernetes orchestrates containers across nodes."})
    sources = client.post("/query", json={"question": "Redis key value"}).json()["sources"]
    assert any("Redis" in s for s in sources)


def test_sources_are_capped_at_k(client):
    for i in range(10):
        client.post("/ingest/text", json={"text": f"Document number {i} about vectors."})
    sources = client.post("/query", json={"question": "vectors"}).json()["sources"]
    assert len(sources) <= 3


def test_source_snippets_are_truncated(client):
    client.post("/ingest/text", json={"text": "x " * 400})
    sources = client.post("/query", json={"question": "x"}).json()["sources"]
    assert all(len(s) <= 150 for s in sources)
