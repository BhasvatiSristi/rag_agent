"""
config/settings.py
Central configuration — loads from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_from_root(value: str) -> str:
	"""Resolve relative paths from project root for consistent cloud/runtime behavior."""
	path = Path(value)
	if path.is_absolute():
		return str(path)
	return str((PROJECT_ROOT / path).resolve())

# --- Storage ---
STORAGE_DIR: str = _resolve_from_root(os.getenv("STORAGE_DIR", "storage"))

# --- ChromaDB ---
CHROMA_PERSIST_DIR: str = _resolve_from_root(
	os.getenv("CHROMA_PERSIST_DIR", "storage/chroma_db")
)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "docs")

# --- BM25 ---
BM25_INDEX_PATH: str = _resolve_from_root(
	os.getenv("BM25_INDEX_PATH", "storage/bm25_index.json")
)

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", 5))
RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "hybrid")
