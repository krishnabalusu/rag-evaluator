import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import Document
from embeddings.embedder import Embedder
from vectorstore.vector_db import VectorStore

TEST_STORE_DIR = "./test_vector_store_temp"


def get_test_store():
    store = VectorStore(persist_directory=TEST_STORE_DIR, collection_name="test_collection")
    store.clear()
    return store


def test_store_and_count():
    embedder = Embedder()
    store = get_test_store()

    docs = [
        Document(text="RAG retrieves relevant documents for answering questions.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
        Document(text="Vector databases store embeddings for fast similarity search.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
        Document(text="ChromaDB is an open source vector database.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
    ]

    docs = embedder.embed_documents(docs)
    store.add_documents(docs)

    assert store.count() == 3
    print(f"  [PASS] Stored and counted {store.count()} chunks")


def test_search_returns_results():
    embedder = Embedder()
    store = get_test_store()

    docs = [
        Document(text="RAG retrieves relevant documents for answering questions.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
        Document(text="Vector databases store embeddings for fast similarity search.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
        Document(text="ChromaDB is an open source vector database.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
    ]

    docs = embedder.embed_documents(docs)
    store.add_documents(docs)

    query_vector = embedder.embed_query("What is a vector database?")
    results = store.search(query_vector, top_k=2)

    assert len(results) == 2
    assert "text" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]
    print(f"  [PASS] Search returned {len(results)} results")
    print(f"         Top result: {results[0]['text'][:60]}...")
    print(f"         Score: {results[0]['score']}")


def test_most_relevant_chunk_ranked_first():
    embedder = Embedder()
    store = get_test_store()

    docs = [
        Document(text="I like eating pizza and pasta.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
        Document(text="Vector databases store high dimensional embeddings.", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
    ]

    docs = embedder.embed_documents(docs)
    store.add_documents(docs)

    query_vector = embedder.embed_query("How are embeddings stored?")
    results = store.search(query_vector, top_k=2)

    assert "vector" in results[0]["text"].lower() or "embed" in results[0]["text"].lower()
    print(f"  [PASS] Most relevant chunk ranked first: {results[0]['text'][:60]}...")


if __name__ == "__main__":
    tests = [
        test_store_and_count,
        test_search_returns_results,
        test_most_relevant_chunk_ranked_first,
    ]

    print("\n" + "=" * 55)
    print("  Step 5: Vector Store Tests")
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