import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.document_loader import Document
from chunking.chunker import chunk_document, chunk_documents


def test_chunks_created():
    doc = Document(text="a" * 2000, metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"})
    chunks = chunk_document(doc, chunk_size=512, overlap=64)
    assert len(chunks) > 1
    print(f"  [PASS] {len(chunks)} chunks created from 2000 char doc")


def test_chunk_size_respected():
    doc = Document(text="a" * 2000, metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"})
    chunks = chunk_document(doc, chunk_size=512, overlap=64)
    for chunk in chunks:
        assert len(chunk.text) <= 512
    print("  [PASS] All chunks within size limit")


def test_metadata_preserved():
    doc = Document(text="a" * 1000, metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"})
    chunks = chunk_document(doc, chunk_size=512, overlap=64)
    for chunk in chunks:
        assert "source" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "total_chunks" in chunk.metadata
    print("  [PASS] Metadata preserved in all chunks")


def test_overlap_works():
    text = "abcdefghij" * 100
    doc = Document(text=text, metadata={"source": "test.txt", "file_name": "test.txt", "doc_type": "txt"})
    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    # chunk 2 should start 80 chars into chunk 1 (overlap of 20)
    assert chunks[0].text[80:100] == chunks[1].text[:20]
    print("  [PASS] Overlap between chunks confirmed")


def test_chunk_documents_multiple():
    docs = [
        Document(text="x" * 1000, metadata={"source": "a.txt", "file_name": "a.txt", "doc_type": "txt"}),
        Document(text="y" * 1000, metadata={"source": "b.txt", "file_name": "b.txt", "doc_type": "txt"}),
    ]
    all_chunks = chunk_documents(docs)
    assert len(all_chunks) > 2
    print(f"  [PASS] {len(all_chunks)} total chunks from 2 documents")


if __name__ == "__main__":
    tests = [
        test_chunks_created,
        test_chunk_size_respected,
        test_metadata_preserved,
        test_overlap_works,
        test_chunk_documents_multiple,
    ]

    print("\n" + "=" * 55)
    print("  Step 3: Chunker Tests")
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