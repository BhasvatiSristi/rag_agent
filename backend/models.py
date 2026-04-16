"""
Purpose:

* Defines request and response data models for the API.

Inputs:

* JSON request fields sent by API clients.

Outputs:

* Validated Python objects for request handling and response serialization.

Used in:

* Imported by FastAPI routes to validate incoming data and shape outgoing data.
"""

from pydantic import BaseModel
from typing import List, Optional
from config.settings import TOP_K, RETRIEVAL_MODE


class AskRequest(BaseModel):
    """
    Input model for the ask endpoint.

    Parameters:

    * question (str): user question text.
    * top_k (int): number of chunks to retrieve.
    * source_file (Optional[str]): optional PDF filename to filter search.
    * retrieval_mode (str): retrieval strategy (hybrid, dense, or bm25).

    Returns:

    * AskRequest: validated request object.

    Steps:

    1. Receive raw JSON fields from the API request.
    2. Validate types and apply default values.
    3. Expose a clean object for route logic.
    """
    question: str
    top_k: int = TOP_K
    source_file: Optional[str] = None
    retrieval_mode: str = RETRIEVAL_MODE


class SourceInfo(BaseModel):
    """
    Metadata model for a source chunk used in the answer.

    Parameters:

    * source (str): source filename.
    * page (int): page number in the source document.
    * score (float): retrieval score for this source.

    Returns:

    * SourceInfo: validated source metadata object.

    Steps:

    1. Receive source details from retrieval results.
    2. Validate each field type.
    3. Keep structured source data for API response.
    """
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    """
    Output model returned by the ask endpoint.

    Parameters:

    * answer (str): generated final answer text.
    * sources (List[SourceInfo]): list of unique source references.
    * chunks_searched (int): number of chunks used for generation.

    Returns:

    * AskResponse: validated response object.

    Steps:

    1. Receive generated answer and source list from route logic.
    2. Validate output fields.
    3. Serialize the response in a stable JSON format.
    """
    answer: str
    sources: List[SourceInfo]
    chunks_searched: int
