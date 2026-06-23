"""
Index Wikipedia articles into vector store.
Run: python index_wiki.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from pipeline.rag_pipeline import RAGPipeline
from vectorstore.vector_db import VectorStore
from config.settings import VECTOR_STORE_DIR, COLLECTION_NAME

WIKI_DOCS_DIR = Path(__file__).parent / "wiki_docs"


def main():
    print("\n" + "=" * 60)
    print("  Indexing Wikipedia Articles into RAG")
    print("=" * 60)

    # Clear old vector store first
    print("\nClearing old vector store...")
    store = VectorStore(persist_directory=str(VECTOR_STORE_DIR), collection_name=COLLECTION_NAME)
    store.clear()
    print("Old data cleared.")

    # Index wiki docs
    pipeline = RAGPipeline()
    total_chunks = pipeline.index(folder=str(WIKI_DOCS_DIR))

    print("=" * 60)
    print(f"  Done! {total_chunks} chunks indexed from {len(list(WIKI_DOCS_DIR.glob('*.txt')))} articles.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
