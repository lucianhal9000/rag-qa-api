import os
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"


class EmptyDocumentError(ValueError):
    """Raised when a source yields no text to index."""


def build_default_embeddings():
    """Load the HuggingFace embedding model.

    Imported lazily: sentence-transformers pulls in torch, which costs several
    seconds to import and ~2GB of wheels to install. Tests inject a fake and
    never pay that price.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_default_llm():
    """Construct the Groq chat model. Lazily imported for the same reason."""
    # Config check first: fail on a missing key before paying for the import.
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add a key "
            "from https://console.groq.com"
        )

    from langchain_groq import ChatGroq

    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=api_key)


class RAGPipeline:
    def __init__(self, embeddings=None, llm=None, k: int = 3):
        # Injectable so tests can substitute deterministic fakes. Production
        # passes nothing and gets the real models.
        self.embeddings = embeddings if embeddings is not None else build_default_embeddings()
        self.llm = llm if llm is not None else build_default_llm()
        self.k = k
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    @property
    def indexed_vectors(self) -> int:
        return self.vectorstore.index.ntotal if self.vectorstore else 0

    def _add(self, docs) -> int:
        if not docs:
            raise EmptyDocumentError("No extractable text found.")
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        self._build_chain()
        return len(docs)

    def ingest_text(self, text: str) -> int:
        docs = self.splitter.create_documents([text])
        count = self._add(docs)
        logger.info(f"Ingested {count} chunks into vector store.")
        return count

    def ingest_file(self, filepath: str) -> int:
        if filepath.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath, encoding="utf-8")
        raw_docs = loader.load()
        docs = self.splitter.split_documents(raw_docs)
        count = self._add(docs)
        logger.info(f"Ingested {count} chunks from file: {filepath}")
        return count

    def _build_chain(self):
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        prompt = PromptTemplate.from_template(
            "Use only the context below to answer the question.\n"
            "If the answer is not in the context, say 'I don't know'.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> dict:
        if not self.chain:
            raise ValueError("No documents ingested yet.")
        answer = self.chain.invoke(question)
        sources = [
            doc.page_content[:150]
            for doc in self.retriever.invoke(question)
        ]
        return {"answer": answer, "sources": sources}
