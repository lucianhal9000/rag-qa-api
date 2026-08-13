import os
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import PromptTemplate                                                 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a key "
                "from https://console.groq.com"
            )
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=api_key,
        )
        self.vectorstore = None
        self.chain = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

    def ingest_text(self, text: str) -> int:
        docs = self.splitter.create_documents([text])
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        self._build_chain()
        logger.info(f"Ingested {len(docs)} chunks into vector store.")
        return len(docs)

    def ingest_file(self, filepath: str) -> int:
        if filepath.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        else:
            loader = TextLoader(filepath)
        raw_docs = loader.load()
        docs = self.splitter.split_documents(raw_docs)
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)
        self._build_chain()
        logger.info(f"Ingested {len(docs)} chunks from file: {filepath}")
        return len(docs)

    def _build_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

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
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        self.retriever = retriever

    def query(self, question: str) -> dict:
        if not self.chain:
            raise ValueError("No documents ingested yet.")
        answer = self.chain.invoke(question)
        sources = [
            doc.page_content[:150]
            for doc in self.retriever.invoke(question)
        ]
        return {"answer": answer, "sources": sources}