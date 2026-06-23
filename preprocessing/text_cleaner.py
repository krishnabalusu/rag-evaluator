import re
import unicodedata


def clean_text(text: str) -> str:
    """Clean raw extracted text for RAG processing."""

    # 1. Normalize unicode (smart quotes, accents, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove page noise patterns like "Page 1 of 10", "Confidential"
    text = re.sub(r'page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'confidential|all rights reserved|copyright', '', text, flags=re.IGNORECASE)

    # 3. Remove special/non-printable characters (keep letters, numbers, punctuation)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    # 4. Replace multiple spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)

    # 5. Replace 3+ blank lines with single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 6. Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(lines)

    # 7. Final strip
    return text.strip()


def clean_documents(documents: list) -> list:
    """Run clean_text on every Document in a list."""
    for doc in documents:
        doc.text = clean_text(doc.text)
    return documents