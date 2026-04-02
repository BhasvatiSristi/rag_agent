"""
retrieval/vectorstore.py
ChromaDB operations: add chunks, query top-k similar chunks.
Supports optional source filtering by filename (for branch-specific search).
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Optional
import os
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K, PROJECT_ROOT
from retrieval.embedder import embed_texts, embed_query


_DEBUG_LOGGED = False


def _log_chroma_debug() -> None:
    global _DEBUG_LOGGED
    if _DEBUG_LOGGED:
        return

    _DEBUG_LOGGED = True
    root_dir = Path(PROJECT_ROOT).resolve()
    storage_dir = root_dir / "storage"
    chroma_dir = root_dir / "storage" / "chroma_db"

    print(f"[chroma-debug] CWD: {os.getcwd()}")
    print(f"[chroma-debug] Root files: {sorted(os.listdir(root_dir))}")
    print(f"[chroma-debug] Persist directory: {chroma_dir}")
    print(f"[chroma-debug] Persist exists: {chroma_dir.exists()}")

    if storage_dir.exists():
        print(f"[chroma-debug] Storage contents: {sorted(os.listdir(storage_dir))}")
    else:
        print(f"[chroma-debug] Storage directory missing: {storage_dir}")

    if chroma_dir.exists():
        print(f"[chroma-debug] Chroma dir contents: {sorted(os.listdir(chroma_dir))}")
    else:
        print(f"[chroma-debug] Chroma directory missing: {chroma_dir}")


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

    # If configured collection is empty, recover by selecting any populated collection.
    for name in _get_collection_names(client):
        if name == CHROMA_COLLECTION_NAME:
            continue
        candidate = client.get_collection(name=name)
        if candidate.count() > 0:
            print(
                f"[chroma-debug] Configured collection '{CHROMA_COLLECTION_NAME}' is empty; "
                f"using populated collection '{name}' instead."
            )
            return candidate

    # For read paths, do not create a new empty collection implicitly.
    if primary is not None:
        return primary
    return None


def _create_client():
    # Prefer chromadb.Client(Settings(...)) with explicit persistence config.
    try:
        return chromadb.Client(
            Settings(
                is_persistent=True,
                persist_directory=CHROMA_PERSIST_DIR,
                anonymized_telemetry=False,
            )
        )
    except TypeError:
        # Fallback for environments that only expose PersistentClient.
        return chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )


def _get_collection(create_if_missing: bool = False):
    _log_chroma_debug()
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
    source_file: Optional[str] = None,   # e.g. "BT-ME.pdf"
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks.
    If source_file is given, only chunks from that file are searched.
    """
    collection = _get_collection(create_if_missing=False)
    if collection is None:
        return []
    q_embedding = embed_query(question)

    # Build where filter if a specific branch file is requested
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


def query_chunks(
    question: str,
    top_k: int = TOP_K,
    source_file: Optional[str] = None,
) -> List[Dict]:
    return query_dense(question, top_k=top_k, source_file=source_file)


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