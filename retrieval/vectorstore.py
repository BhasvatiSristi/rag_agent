"""
Purpose:

* Handles ChromaDB storage and dense vector retrieval.

Inputs:

* Chunk dictionaries for storage, and question text for retrieval.

Outputs:

* Stored vectors in ChromaDB and ranked retrieval results.

Used in:

* Called during ingestion to add chunks.
* Called during question answering to fetch relevant context.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K
from retrieval.embedder import embed_texts, embed_query

def _get_collection_names(client) -> List[str]:
    """
    Read available collection names from a Chroma client.

    Parameters:

    * client: active Chroma client instance.

    Returns:

    * List[str]: collection names that can be queried.

    Steps:

    1. Request collection list from the client.
    2. Handle errors by returning an empty list.
    3. Extract names from string or object variants.
    4. Return normalized name list.
    """
    names: List[str] = []
    try:
        collections = client.list_collections()
    except Exception:
        return names

    for item in collections:
        if isinstance(item, str):
            names.append(item)
        else:
            name = getattr(item, "name", None)
            if name:
                names.append(name)
    return names


def _resolve_collection(client, create_if_missing: bool):
    """
    Pick the best collection to use for retrieval or storage.

    Parameters:

    * client: active Chroma client instance.
    * create_if_missing (bool): whether to create the main collection if needed.

    Returns:

    * Collection or None: selected collection object.

    Steps:

    1. Try to open or create the primary configured collection.
    2. If primary has data, use it.
    3. If primary is empty, scan other collections for existing data.
    4. Return the best available collection.
    """
    if create_if_missing:
        primary = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        try:
            primary = client.get_collection(name=CHROMA_COLLECTION_NAME)
        except Exception:
            primary = None

    if primary is not None and primary.count() > 0:
        return primary

    for name in _get_collection_names(client):
        if name == CHROMA_COLLECTION_NAME:
            continue
        candidate = client.get_collection(name=name)
        if candidate.count() > 0:
            print(f"⚠️  Chroma collection '{CHROMA_COLLECTION_NAME}' is empty; using '{name}' instead.")
            return candidate

    return primary


def _create_client():
    """
    Create a Chroma client compatible with different SDK versions.

    Parameters:

    * None

    Returns:

    * Chroma client instance.

    Steps:

    1. Try the modern Client + Settings constructor.
    2. If that fails, fallback to PersistentClient style.
    3. Return a working client object.
    """
    try:
        return chromadb.Client(
            Settings(
                is_persistent=True,
                persist_directory=CHROMA_PERSIST_DIR,
                anonymized_telemetry=False,
            )
        )
    except TypeError:
        return chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )


def _get_collection(create_if_missing: bool = False):
    """
    Get a usable Chroma collection from the persistence directory.

    Parameters:

    * create_if_missing (bool): create configured collection when missing.

    Returns:

    * Collection or None: resolved collection for operations.

    Steps:

    1. Ensure persist directory exists.
    2. Create a Chroma client.
    3. Resolve and return the best collection.
    """
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    client = _create_client()
    collection = _resolve_collection(client, create_if_missing=create_if_missing)
    return collection


def add_chunks(chunks: List[Dict]) -> None:
    """
    Add new chunk embeddings into ChromaDB.

    Parameters:

    * chunks (List[Dict]): chunk records with text and metadata.

    Returns:

    * None

    Steps:

    1. Open or create the target collection.
    2. Skip chunks that already exist by chunk_id.
    3. Embed only new chunk texts.
    4. Store ids, embeddings, documents, and metadata.
    """
    collection = _get_collection(create_if_missing=True)
    existing = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        print("  → All chunks already in ChromaDB. Skipping ingestion.")
        return

    print(f"  → Embedding {len(new_chunks)} new chunks...")
    texts = [c["text"] for c in new_chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c["chunk_id"] for c in new_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in new_chunks],
    )
    print(f"  → Stored {len(new_chunks)} chunks in ChromaDB ✅")


def query_dense(
    question: str,
    top_k: int = TOP_K,
    source_file: Optional[str] = None,
) -> List[Dict]:
    """
    Retrieve top matching chunks using dense vector similarity.

    Parameters:

    * question (str): user query text.
    * top_k (int): maximum number of chunks to return.
    * source_file (Optional[str]): optional filename filter.

    Returns:

    * List[Dict]: ranked chunk results with text, source, page, and score.

    Steps:

    1. Load existing collection.
    2. Embed the user question.
    3. Apply optional source filter.
    4. Query Chroma for nearest chunks.
    5. Convert raw results to response-friendly dictionaries.
    """
    collection = _get_collection(create_if_missing=False)
    if collection is None:
        return []
    q_embedding = embed_query(question)

    where = {"source": source_file} if source_file else None

    total = collection.count()
    if total == 0:
        return []

    kwargs = dict(
        query_embeddings=[q_embedding],
        n_results=min(top_k, total),
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "page": meta.get("page", 0),
            "score": round(1 - dist, 4),
        })

    return chunks


def collection_size() -> int:
    """
    Return number of stored chunks in the selected collection.

    Parameters:

    * None

    Returns:

    * int: count of chunks, or 0 when collection is missing.

    Steps:

    1. Open collection without creating a new one.
    2. If unavailable, return 0.
    3. Otherwise return collection count.
    """
    collection = _get_collection(create_if_missing=False)
    return collection.count() if collection is not None else 0


def list_sources() -> List[str]:
    """
    Return sorted unique source filenames from stored metadata.

    Parameters:

    * None

    Returns:

    * List[str]: sorted source names currently present in the collection.

    Steps:

    1. Open collection if available.
    2. Read all metadata records.
    3. Extract source names.
    4. Remove duplicates and return sorted output.
    """
    collection = _get_collection(create_if_missing=False)
    if collection is None:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted(set(m["source"] for m in all_meta if "source" in m))
    return sources