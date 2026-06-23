"""
Step 1 Tests — Document Loader

Run:  python -m pytest tests/test_loader.py -v
  or: python tests/test_loader.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from ingestion.document_loader import load_document, load_directory, Document

SAMPLE_DIR = Path(__file__).parent.parent / "sample_docs"


# ── Helpers ────────────────────────────────────────────────────────────────

def _assert_doc(docs, min_chars=50):
    assert isinstance(docs, list), "load_document must return a list"
    assert len(docs) > 0, "Got zero documents"
    for doc in docs:
        assert isinstance(doc, Document), f"Expected Document, got {type(doc)}"
        assert len(doc.text) >= min_chars, f"Text too short: {len(doc.text)} chars"
        assert "source" in doc.metadata, "Missing 'source' in metadata"
        assert "file_name" in doc.metadata, "Missing 'file_name' in metadata"
        assert "doc_type" in doc.metadata, "Missing 'doc_type' in metadata"


# ── Tests ──────────────────────────────────────────────────────────────────

def test_load_txt():
    path = SAMPLE_DIR / "ai_overview.txt"
    docs = load_document(path)
    _assert_doc(docs)
    assert docs[0].metadata["doc_type"] == "txt"
    print(f"  [PASS] TXT: {len(docs[0].text)} chars")


def test_load_markdown():
    path = SAMPLE_DIR / "company_policy.md"
    docs = load_document(path)
    _assert_doc(docs)
    assert docs[0].metadata["doc_type"] == "markdown"
    print(f"  [PASS] Markdown: {len(docs[0].text)} chars")


def test_load_html():
    path = SAMPLE_DIR / "sample.html"
    docs = load_document(path)
    _assert_doc(docs)
    assert docs[0].metadata["doc_type"] == "html"
    # HTML loader strips tags — should NOT contain raw HTML
    assert "<html" not in docs[0].text.lower(), "HTML tags leaked into extracted text"
    print(f"  [PASS] HTML: {len(docs[0].text)} chars (tags stripped)")


def test_load_directory():
    docs = load_directory(SAMPLE_DIR)
    assert len(docs) >= 3, f"Expected at least 3 docs, got {len(docs)}"
    # All docs have required metadata
    for doc in docs:
        assert doc.text.strip(), "Empty document in results"
        assert doc.metadata.get("source"), "Missing source metadata"
    print(f"  [PASS] Directory: {len(docs)} total docs loaded")


def test_metadata_completeness():
    path = SAMPLE_DIR / "ai_overview.txt"
    docs = load_document(path)
    meta = docs[0].metadata
    required_keys = {"source", "file_name", "doc_type"}
    missing = required_keys - set(meta.keys())
    assert not missing, f"Missing metadata keys: {missing}"
    print(f"  [PASS] Metadata keys present: {list(meta.keys())}")


def test_unsupported_extension():
    try:
        load_document(SAMPLE_DIR / "fake.xyz")
        assert False, "Should have raised ValueError"
    except (ValueError, FileNotFoundError):
        print("  [PASS] Unsupported extension raises correct error")


def test_file_not_found():
    try:
        load_document("/nonexistent/path/file.txt")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  [PASS] Missing file raises FileNotFoundError")


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_load_txt,
        test_load_markdown,
        test_load_html,
        test_load_directory,
        test_metadata_completeness,
        test_unsupported_extension,
        test_file_not_found,
    ]

    print("\n" + "=" * 55)
    print("  Step 1: Document Loader Tests")
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

    if failed:
        sys.exit(1)
