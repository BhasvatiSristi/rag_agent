"""
config/settings.py
Central configuration — loads from .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Storage ---
STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")

# --- ChromaDB ---
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./storage/chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "curriculum")

# --- BM25 ---
BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "./storage/bm25_index.json")

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", 5))
RETRIEVAL_MODE: str = os.getenv("RETRIEVAL_MODE", "hybrid")
