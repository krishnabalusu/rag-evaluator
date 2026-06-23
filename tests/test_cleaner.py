import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_cleaner import clean_text, clean_documents
from ingestion.document_loader import Document


def test_removes_extra_spaces():
    result = clean_text("this   is    messy   text")
    assert result == "this is messy text"
    print("  [PASS] Extra spaces removed")


def test_removes_extra_blank_lines():
    result = clean_text("line1\n\n\n\n\nline2")
    assert "\n\n\n" not in result
    print("  [PASS] Extra blank lines removed")


def test_removes_page_noise():
    result = clean_text("some text Page 1 of 10 more text")
    assert "Page 1 of 10" not in result
    print("  [PASS] Page noise removed")


def test_unicode_normalized():
    result = clean_text("\u201chello\u201d")  # smart quotes
    assert "\u201c" not in result
    print("  [PASS] Unicode normalized")


def test_clean_documents():
    docs = [Document(text="hello   world\n\n\n\ntest", metadata={})]
    cleaned = clean_documents(docs)
    assert "   " not in cleaned[0].text
    print("  [PASS] clean_documents works on list")


if __name__ == "__main__":
    tests = [
        test_removes_extra_spaces,
        test_removes_extra_blank_lines,
        test_removes_page_noise,
        test_unicode_normalized,
        test_clean_documents,
    ]

    print("\n" + "=" * 55)
    print("  Step 2: Text Cleaner Tests")
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