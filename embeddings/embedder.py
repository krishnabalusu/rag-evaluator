import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from sentence_transformers import SentenceTransformer
from ingestion.document_loader import Document
from config.settings import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        """Convert single text string to embedding vector."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, documents: List[Document]) -> List[Document]:
        """Add embedding to each Document's metadata."""
        texts = [doc.text for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        for doc, embedding in zip(documents, embeddings):
            doc.metadata["embedding"] = embedding.tolist()

        return documents

    def embed_query(self, query: str) -> List[float]:
        """Embed a user query — same model as documents."""
        return self.embed_text(query)