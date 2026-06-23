"""
Central configuration for the RAG system.
All tuneable parameters live here — change once, applies everywhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "sample_docs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store_data"

# ── Document ingestion ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".html", ".htm", ".md"}

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE = 512          # characters per chunk
CHUNK_OVERLAP = 64        # overlap between consecutive chunks

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # free, fast, good quality

# ── Vector store ───────────────────────────────────────────────────────────
COLLECTION_NAME = "enterprise_rag"
TOP_K = 5                 # chunks returned per query

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")   # "anthropic" | "openai"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.0    # deterministic — enterprise grounding requirement
