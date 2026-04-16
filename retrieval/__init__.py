"""
Purpose:

* Exposes retrieval functions from dense, BM25, and hybrid modules.

Inputs:

* No direct runtime inputs in this file.

Outputs:

* Re-exported retrieval functions for simpler imports.

Used in:

* API routes and scripts that need retrieval utilities.
"""

from .hybrid import hybrid_query
from .vectorstore import query_dense
from .bm25 import query_bm25, build_bm25_index

__all__ = [
    "hybrid_query",
    "query_dense",
    "query_bm25",
    "build_bm25_index",
]
