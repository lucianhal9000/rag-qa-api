"""Test fixtures.

The real pipeline loads a sentence-transformers model and talks to Groq. Both
are replaced here so the suite runs offline, needs no API key, and finishes in
seconds. Everything else — FAISS, the text splitter, the LCEL chain, FastAPI
routing — is the real thing.
"""
import hashlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app, get_rag  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402

DIM = 64
FAKE_ANSWER = "This is a fake answer."


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embedding.

    Random fake vectors would let the chain run but make retrieval meaningless,
    so retrieval assertions would prove nothing. Hashing each token into a fixed
    bucket means documents sharing words land near each other, which is enough
    to assert that the right chunk comes back.
    """

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in text.lower().split():
            digest = hashlib.md5(token.encode()).digest()
            vec[digest[0] % DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def fake_llm():
    """Stands in for ChatGroq. StrOutputParser accepts a plain string."""
    return RunnableLambda(lambda _prompt_value: FAKE_ANSWER)


@pytest.fixture
def pipeline() -> RAGPipeline:
    return RAGPipeline(embeddings=HashingEmbeddings(), llm=fake_llm())


@pytest.fixture
def client(pipeline):
    app.dependency_overrides[get_rag] = lambda: pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
