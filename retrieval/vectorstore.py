"""
retrieval/vectorstore.py
ChromaDB operations: add chunks, query top-k similar chunks.
Supports optional source filtering by filename (for branch-specific search).
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K
from retrieval.embedder import embed_texts, embed_query

def _get_collection_names(client) -> List[str]:
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
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    client = _create_client()
    collection = _resolve_collection(client, create_if_missing=create_if_missing)
    return collection


def add_chunks(chunks: List[Dict]) -> None:
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
    collection = _get_collection(create_if_missing=False)
    return collection.count() if collection is not None else 0


def list_sources() -> List[str]:
    """Return all unique source filenames stored in ChromaDB."""
    collection = _get_collection(create_if_missing=False)
    if collection is None:
        return []
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted(set(m["source"] for m in all_meta if "source" in m))
    return sources