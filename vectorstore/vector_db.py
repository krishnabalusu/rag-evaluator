import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from ingestion.document_loader import Document
from config.settings import VECTOR_STORE_DIR, COLLECTION_NAME, TOP_K


class VectorStore:
    def __init__(self, persist_directory: str = str(VECTOR_STORE_DIR), collection_name: str = COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine similarity
        )
        print(f"Vector store ready: {persist_directory} | collection: {collection_name}")

    def add_documents(self, documents: List[Document]) -> None:
        """Store chunks with their embeddings in ChromaDB."""
        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            embedding = doc.metadata.get("embedding")
            if embedding is None:
                raise ValueError(f"Document {i} has no embedding. Run Embedder first.")

            # ChromaDB metadata must be flat (str/int/float only)
            meta = {k: str(v) for k, v in doc.metadata.items() if k != "embedding"}

            ids.append(f"chunk_{i}_{meta.get('source', 'unknown')}_{meta.get('chunk_index', i)}")
            embeddings.append(embedding)
            texts.append(doc.text)
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        print(f"Stored {len(documents)} chunks in vector store.")

    def search(self, query_embedding: List[float], top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Find top-k most similar chunks to a query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for text, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            hits.append({
                "text": text,
                "metadata": meta,
                "score": round(1 - distance, 4)  # convert distance to similarity score
            })

        return hits

    def count(self) -> int:
        """Return number of chunks stored."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all chunks from collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print("Vector store cleared.")