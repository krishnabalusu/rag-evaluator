import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import Document
from embeddings.embedder import Embedder
from vectorstore.vector_db import VectorStore
from retrieval.retriever import Retriever

TEST_STORE_DIR = "./test_retriever_temp"


def setup():
    embedder = Embedder()
    store = VectorStore(persist_directory=TEST_STORE_DIR, collection_name="test_retriever")
    store.clear()

    docs = [
        Document(text="RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating answers.", metadata={"source": "ai.txt", "file_name": "ai.txt", "doc_type": "txt"}),
        Document(text="ChromaDB is a vector database used to store and search embeddings efficiently.", metadata={"source": "ai.txt", "file_name": "ai.txt", "doc_type": "txt"}),
        Document(text="Remote work policy requires VPN and MFA for all company systems.", metadata={"source": "policy.txt", "file_name": "policy.txt", "doc_type": "txt"}),
        Document(text="Employees must submit expenses within 30 days via Concur.", metadata={"source": "policy.txt", "file_name": "policy.txt", "doc_type": "txt"}),
        Document(text="Large language models use transformer architecture with self-attention mechanism.", metadata={"source": "ai.txt", "file_name": "ai.txt", "doc_type": "txt"}),
    ]

    docs = embedder.embed_documents(docs)
    store.add_documents(docs)
    retriever = Retriever(embedder, store, top_k=3)
    return retriever


def test_retrieve_returns_results():
    retriever = setup()
    results = retriever.retrieve("What is RAG?")
    assert len(results) > 0
    assert "text" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]
    print(f"  [PASS] Retrieved {len(results)} results")


def test_relevant_chunk_ranked_first():
    retriever = setup()
    results = retriever.retrieve("What is RAG?")
    top_text = results[0]["text"].lower()
    assert "rag" in top_text or "retrieval" in top_text
    print(f"  [PASS] Top result relevant: {results[0]['text'][:60]}...")


def test_format_context():
    retriever = setup()
    results = retriever.retrieve("vector database")
    context = retriever.format_context(results)
    assert "Source 1" in context
    assert "score" in context
    assert "chunk" in context
    print(f"  [PASS] Context formatted correctly")
    print(f"         Preview:\n{context[:200]}...")


def test_no_results_handled():
    retriever = setup()
    # Very low score query — format_context should handle gracefully
    results = []
    context = retriever.format_context(results)
    assert context == "No relevant context found."
    print("  [PASS] Empty results handled gracefully")


if __name__ == "__main__":
    tests = [
        test_retrieve_returns_results,
        test_relevant_chunk_ranked_first,
        test_format_context,
        test_no_results_handled,
    ]

    print("\n" + "=" * 55)
    print("  Step 6: Retriever Tests")
    print("=" * 55)

    passed, failed = 0, 0
    for test in tests:
        print(f"\n[RUN] {test.__name__}")
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("\n" + "=" * 55)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 55 + "\n")