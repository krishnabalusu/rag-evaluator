import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import Document
from embeddings.embedder import Embedder


def test_embed_text_returns_vector():
    embedder = Embedder()
    vector = embedder.embed_text("What is RAG?")
    assert isinstance(vector, list)
    assert len(vector) == 384
    print(f"  [PASS] Vector length: {len(vector)}")


def test_similar_texts_close():
    embedder = Embedder()
    v1 = embedder.embed_text("What is RAG?")
    v2 = embedder.embed_text("Explain retrieval augmented generation")
    v3 = embedder.embed_text("I like eating pizza")

    # dot product similarity
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    sim_related = dot(v1, v2)
    sim_unrelated = dot(v1, v3)

    assert sim_related > sim_unrelated
    print(f"  [PASS] Related texts closer ({sim_related:.2f}) than unrelated ({sim_unrelated:.2f})")


def test_embed_documents():
    embedder = Embedder()
    docs = [
        Document(text="RAG retrieves documents", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
        Document(text="Embeddings are vectors", metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"}),
    ]
    docs = embedder.embed_documents(docs)
    for doc in docs:
        assert "embedding" in doc.metadata
        assert len(doc.metadata["embedding"]) == 384
    print(f"  [PASS] {len(docs)} documents embedded")


def test_embed_query():
    embedder = Embedder()
    vector = embedder.embed_query("What is a vector database?")
    assert isinstance(vector, list)
    assert len(vector) == 384
    print(f"  [PASS] Query embedded: {len(vector)} dims")


if __name__ == "__main__":
    tests = [
        test_embed_text_returns_vector,
        test_similar_texts_close,
        test_embed_documents,
        test_embed_query,
    ]

    print("\n" + "=" * 55)
    print("  Step 4: Embedder Tests")
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