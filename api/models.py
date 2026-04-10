"""
api/models.py
Pydantic models for request/response validation.
"""

from pydantic import BaseModel
from typing import List, Optional
from config.settings import TOP_K, RETRIEVAL_MODE


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str
    top_k: int = TOP_K
    source_file: Optional[str] = None
    retrieval_mode: str = RETRIEVAL_MODE


class SourceInfo(BaseModel):
    """Source document metadata for answer."""
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str
    sources: List[SourceInfo]
    chunks_searched: int
