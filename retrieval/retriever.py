import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any
from embeddings.embedder import Embedder
from vectorstore.vector_db import VectorStore
from config.settings import TOP_K


class Retriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = TOP_K):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Given a user query, return top-k relevant chunks.
        Each result has: text, score, metadata.
        """
        k = top_k or self.top_k

        # Step 1: embed the query
        query_embedding = self.embedder.embed_query(query)

        # Step 2: search vector store
        results = self.vector_store.search(query_embedding, top_k=k)

        # Step 3: filter out very low scoring results
        results = [r for r in results if r["score"] > 0.1]

        return results

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a single context string for the LLM."""
        if not results:
            return "No relevant context found."

        context_parts = []
        for i, result in enumerate(results, start=1):
            source = result["metadata"].get("file_name", "unknown")
            chunk_idx = result["metadata"].get("chunk_index", "?")
            score = result["score"]
            text = result["text"]

            context_parts.append(
                f"[Source {i}: {source} | chunk {chunk_idx} | score {score}]\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)