"""
retrieval/embedder.py
Vectorizes text using Cohere API embeddings (embed-english-v3.0).
API-based approach: lightweight, fast cold starts, suitable for cloud deployment.
"""

import os
from typing import List
import cohere

# Initialize Cohere client once at module level
_client: cohere.Client | None = None


def get_client() -> cohere.Client:
    """Get or create Cohere API client (singleton pattern)."""
    global _client
    if _client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "COHERE_API_KEY environment variable not set. "
                "Get a free key from https://cohere.com/"
            )
        _client = cohere.Client(api_key=api_key)
        print("  → Cohere API client initialized ✅")
    return _client


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of document chunks using Cohere API.
    input_type='search_document' optimizes for document retrieval.
    Returns list of 1024-dim embeddings as floats.
    """
    if not texts:
        return []
    
    client = get_client()
    response = client.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )
    return response.embeddings["float"]


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string using Cohere API.
    input_type='search_query' optimizes for query embedding (different from documents).
    Returns 1024-dim embedding as float list.
    """
    client = get_client()
    response = client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    return response.embeddings["float"][0]
