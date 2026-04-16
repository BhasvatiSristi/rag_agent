"""
Purpose:

* Stores central configuration values for the whole project.

Inputs:

* Environment variables from the system and .env file.

Outputs:

* Module-level constants used by ingestion, retrieval, generation, and API layers.

Used in:

* Imported by many modules to keep configuration in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_from_root(value: str) -> str:
	"""
	Convert a path value to an absolute path based on the project root.

	Parameters:

	* value (str): relative or absolute path string from environment/config.

	Returns:

	* str: absolute path string.

	Steps:

	1. Build a Path object from the input.
	2. If already absolute, return it directly.
	3. If relative, join it with PROJECT_ROOT.
	4. Return the resolved absolute path.
	"""
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
