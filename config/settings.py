"""
config/settings.py
Central configuration — loads from .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Ollama ---
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en")

# --- ChromaDB ---
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "curriculum")

# --- Chunking ---
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 700))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", 5))
