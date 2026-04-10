"""
retrieval/hybrid.py
Hybrid retrieval: dense Chroma + BM25 merged with reciprocal rank fusion (RRF).
"""

from typing import Dict, List, Optional

from retrieval.vectorstore import query_dense
from retrieval.bm25 import query_bm25

RRF_K = 60


def _chunk_key(chunk: Dict) -> str:
    return f"{chunk.get('source', 'unknown')}::{chunk.get('page', 0)}::{hash(chunk.get('text', ''))}"


def _rrf_merge(dense_chunks: List[Dict], bm25_chunks: List[Dict], top_k: int) -> List[Dict]:
    merged: Dict[str, Dict] = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        key = _chunk_key(chunk)
        if key not in merged:
            merged[key] = dict(chunk)
            merged[key]["rrf_score"] = 0.0
        merged[key]["rrf_score"] += 1.0 / (RRF_K + rank)

    for rank, chunk in enumerate(bm25_chunks, start=1):
        key = _chunk_key(chunk)
        if key not in merged:
            merged[key] = dict(chunk)
            merged[key]["rrf_score"] = 0.0
        merged[key]["rrf_score"] += 1.0 / (RRF_K + rank)

    ranked = sorted(merged.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)

    out = []
    for item in ranked[:top_k]:
        item["score"] = round(item.get("rrf_score", 0.0), 4)
        out.append(item)
    return out


def hybrid_query(
    question: str,
    top_k: int = 5,
    source_file: Optional[str] = None,
    mode: str = "hybrid",
) -> List[Dict]:
    mode_norm = (mode or "hybrid").strip().lower()

    if mode_norm == "dense":
        return query_dense(question, top_k=top_k, source_file=source_file)

    if mode_norm == "bm25":
        return query_bm25(question, top_k=top_k, source_file=source_file)

    candidate_k = max(top_k * 2, 8)
    dense_chunks = query_dense(question, top_k=candidate_k, source_file=source_file)
    bm25_chunks = query_bm25(question, top_k=candidate_k, source_file=source_file)
    return _rrf_merge(dense_chunks, bm25_chunks, top_k=top_k)
