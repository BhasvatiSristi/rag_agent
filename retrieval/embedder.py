"""
retrieval/embedder.py
Wraps SentenceTransformers for encoding text into vectors.
Model: BAAI/bge-small-en (fast, good quality, ~130MB)
"""

from sentence_transformers import SentenceTransformer
from typing import List
from config.settings import EMBEDDING_MODEL

# Load once at module level (cached after first import)
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"  → Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"  → Embedding model loaded ✅")
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings.
    Returns list of float vectors.
    """
    model = get_model()
    # normalize_embeddings=True improves cosine similarity quality for BGE models
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Embed a single query string.
    BGE models recommend prepending "Represent this sentence for searching relevant passages: "
    for query embedding (not for document embedding).
    """
    model = get_model()
    prefixed = f"Represent this sentence for searching relevant passages: {query}"
    embedding = model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)
    return embedding[0].tolist()
