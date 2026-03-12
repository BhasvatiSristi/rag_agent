"""
retrieval/vectorstore.py
ChromaDB operations: add chunks, query top-k similar chunks.
Supports optional source filtering by filename (for branch-specific search).
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K
from retrieval.embedder import embed_texts, embed_query


def _get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def add_chunks(chunks: List[Dict]) -> None:
    collection = _get_collection()
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


def query_chunks(
    question: str,
    top_k: int = TOP_K,
    source_file: Optional[str] = None,   # e.g. "BT-ME.pdf"
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks.
    If source_file is given, only chunks from that file are searched.
    """
    collection = _get_collection()
    q_embedding = embed_query(question)

    # Build where filter if a specific branch file is requested
    where = {"source": source_file} if source_file else None

    kwargs = dict(
        query_embeddings=[q_embedding],
        n_results=min(top_k, collection.count()),
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
    return _get_collection().count()


def list_sources() -> List[str]:
    """Return all unique source filenames stored in ChromaDB."""
    collection = _get_collection()
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted(set(m["source"] for m in all_meta if "source" in m))
    return sources