"""Unit tests against RAGPipeline directly, bypassing HTTP."""
import pytest

from rag_pipeline import EmptyDocumentError, RAGPipeline, build_default_llm


def test_pipeline_starts_empty(pipeline):
    assert pipeline.vectorstore is None
    assert pipeline.indexed_vectors == 0
    assert pipeline.chain is None


def test_query_without_documents_raises(pipeline):
    with pytest.raises(ValueError, match="No documents ingested"):
        pipeline.query("anything")


def test_ingest_builds_the_chain(pipeline):
    pipeline.ingest_text("Vector databases store embeddings.")
    assert pipeline.chain is not None
    assert pipeline.retriever is not None


def test_empty_document_set_raises_typed_error(pipeline):
    with pytest.raises(EmptyDocumentError):
        pipeline._add([])


def test_k_is_configurable():
    from conftest import HashingEmbeddings, fake_llm

    p = RAGPipeline(embeddings=HashingEmbeddings(), llm=fake_llm(), k=1)
    for i in range(5):
        p.ingest_text(f"Sentence {i} about retrieval.")
    assert len(p.query("retrieval")["sources"]) == 1


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_default_llm()
