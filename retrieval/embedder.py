"""
Purpose:

* Creates vector embeddings for documents and user queries using Cohere.

Inputs:

* Text chunks and query strings.

Outputs:

* Float embedding vectors used by dense retrieval.

Used in:

* Called by vectorstore functions during ingestion and query time.
"""

import os
from typing import List, Optional
import cohere

# Initialize Cohere client once at module level
_client: Optional[cohere.Client] = None


def _extract_float_embeddings(response) -> List[List[float]]:
    """
    Extract float embeddings from different Cohere SDK response formats.

    Parameters:

    * response: Cohere embed API response object.

    Returns:

    * List[List[float]]: list of float embedding vectors.

    Steps:

    1. Read embeddings field from the response.
    2. Try typed response format first.
    3. Fallback to dict format for compatibility.
    4. Raise error if format is unsupported.
    """
    embeddings = response.embeddings

    # Cohere v5 typed response: response.embeddings.float
    float_vectors = getattr(embeddings, "float", None)
    if float_vectors is not None:
        return float_vectors

    # Backward-compatible fallback for dict-like responses.
    if isinstance(embeddings, dict) and "float" in embeddings:
        return embeddings["float"]

    raise TypeError("Unsupported Cohere embeddings response format")


def get_client() -> cohere.Client:
    """
    Get a shared Cohere client instance.

    Parameters:

    * None

    Returns:

    * cohere.Client: initialized API client.

    Steps:

    1. Reuse cached client if already created.
    2. Read API key from environment.
    3. Raise a clear error if key is missing.
    4. Create and cache client for later calls.
    """
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
    Embed document chunks for dense retrieval.

    Parameters:

    * texts (List[str]): chunk texts to embed.

    Returns:

    * List[List[float]]: embedding vectors for all input texts.

    Steps:

    1. Return empty list for empty input.
    2. Get initialized Cohere client.
    3. Call Cohere embed API with search_document input type.
    4. Extract and return float vectors.
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
    return _extract_float_embeddings(response)


def embed_query(query: str) -> List[float]:
    """
    Embed one user query string for dense retrieval.

    Parameters:

    * query (str): question text from user.

    Returns:

    * List[float]: query embedding vector.

    Steps:

    1. Get initialized Cohere client.
    2. Call embed API using search_query input type.
    3. Extract first embedding vector and return it.
    """
    client = get_client()
    response = client.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    return _extract_float_embeddings(response)[0]
