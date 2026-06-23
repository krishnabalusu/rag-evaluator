import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from ingestion.document_loader import Document
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_document(doc: Document, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Split one Document into smaller overlapping chunks."""
    text = doc.text
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_meta = doc.metadata.copy()
            chunk_meta["chunk_index"] = chunk_index
            chunk_meta["start_char"] = start
            chunk_meta["end_char"] = end

            chunks.append(Document(text=chunk_text, metadata=chunk_meta))
            chunk_index += 1

        start = end - overlap  # move forward with overlap

    # store total chunks in metadata
    for chunk in chunks:
        chunk.metadata["total_chunks"] = len(chunks)

    return chunks


def chunk_documents(documents: List[Document], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Chunk all documents in a list."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
    return all_chunks