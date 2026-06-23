import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import load_directory
from preprocessing.text_cleaner import clean_documents
from chunking.chunker import chunk_documents
from embeddings.embedder import Embedder
from vectorstore.vector_db import VectorStore
from retrieval.retriever import Retriever
from generation.generator import generate_answer
from config.settings import SAMPLE_DOCS_DIR, VECTOR_STORE_DIR, COLLECTION_NAME


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore(
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=COLLECTION_NAME
        )
        self.retriever = Retriever(self.embedder, self.vector_store)

    def index(self, folder: str = str(SAMPLE_DOCS_DIR)) -> int:
        """Load, clean, chunk, embed and store all documents from folder."""
        print(f"\n>>> Indexing documents from: {folder}")

        docs = load_directory(folder)
        print(f"    Loaded: {len(docs)} documents")

        docs = clean_documents(docs)
        print(f"    Cleaned: {len(docs)} documents")

        chunks = chunk_documents(docs)
        print(f"    Chunked: {len(chunks)} chunks")

        chunks = self.embedder.embed_documents(chunks)
        print(f"    Embedded: {len(chunks)} chunks")

        self.vector_store.add_documents(chunks)
        print(f"    Stored: {len(chunks)} chunks in vector store")
        print(f">>> Indexing complete.\n")

        return len(chunks)

    def query(self, question: str) -> dict:
        """Take user question -> retrieve context -> generate answer."""
        print(f"\n>>> Question: {question}")

        results = self.retriever.retrieve(question)
        context = self.retriever.format_context(results)

        if not results:
            return {
                "question": question,
                "answer": "I don't have enough information to answer this question.",
                "sources": [],
            }

        response = generate_answer(question, context)

        sources = []
        for r in results:
            source = {
                "file": r["metadata"].get("file_name", "unknown"),
                "chunk": r["metadata"].get("chunk_index", "?"),
                "score": r["score"],
            }
            if source not in sources:
                sources.append(source)

        return {
            "question": question,
            "answer": response["answer"],
            "sources": sources,
            "tokens_used": response["tokens_used"],
            "model": response["model"],
        }
