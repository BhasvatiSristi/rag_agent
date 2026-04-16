"""
Purpose:

* Combines dense and BM25 retrieval into one hybrid result list.

Inputs:

* User question text, retrieval mode, top_k, and optional source filter.

Outputs:

* Ranked chunk list from dense, BM25, or fused hybrid retrieval.

Used in:

* Called by API and test script as the main retrieval entry point.
"""

from typing import Dict, List, Optional

from retrieval.vectorstore import query_dense
from retrieval.bm25 import query_bm25

RRF_K = 60


def _chunk_key(chunk: Dict) -> str:
    """
    Create a stable key for identifying a chunk during merge.

    Parameters:

    * chunk (Dict): chunk data dictionary.

    Returns:

    * str: unique key based on source, page, and text hash.

    Steps:

    1. Read source, page, and text fields.
    2. Build a combined string key.
    3. Return the key for deduplication.
    """
    return f"{chunk.get('source', 'unknown')}::{chunk.get('page', 0)}::{hash(chunk.get('text', ''))}"


def _rrf_merge(dense_chunks: List[Dict], bm25_chunks: List[Dict], top_k: int) -> List[Dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion (RRF).

    Parameters:

    * dense_chunks (List[Dict]): chunks from dense retrieval.
    * bm25_chunks (List[Dict]): chunks from BM25 retrieval.
    * top_k (int): maximum number of merged results.

    Returns:

    * List[Dict]: fused top-ranked chunk list.

    Steps:

    1. Add dense results with rank-based RRF contribution.
    2. Add BM25 results with the same RRF rule.
    3. Merge duplicates by chunk key.
    4. Sort by total RRF score.
    5. Return top_k items.
    """
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
    """
    Retrieve chunks using dense, BM25, or hybrid mode.

    Parameters:

    * question (str): user query text.
    * top_k (int): number of final results.
    * source_file (Optional[str]): optional source filename filter.
    * mode (str): retrieval mode (dense, bm25, or hybrid).

    Returns:

    * List[Dict]: ranked chunk list based on selected mode.

    Steps:

    1. Normalize retrieval mode text.
    2. Route directly for dense or BM25 modes.
    3. For hybrid mode, fetch candidates from both retrievers.
    4. Fuse candidates with RRF.
    5. Return top results.
    """
    mode_norm = (mode or "hybrid").strip().lower()

    if mode_norm == "dense":
        return query_dense(question, top_k=top_k, source_file=source_file)

    if mode_norm == "bm25":
        return query_bm25(question, top_k=top_k, source_file=source_file)

    candidate_k = max(top_k * 2, 8)
    dense_chunks = query_dense(question, top_k=candidate_k, source_file=source_file)
    bm25_chunks = query_bm25(question, top_k=candidate_k, source_file=source_file)
    return _rrf_merge(dense_chunks, bm25_chunks, top_k=top_k)
