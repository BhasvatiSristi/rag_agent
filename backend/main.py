"""
Purpose:

* Defines the FastAPI application and HTTP endpoints.

Inputs:

* HTTP requests from clients (question text, optional source file, retrieval settings).

Outputs:

* JSON responses for API metadata, health status, and generated answers.

Used in:

* Started by the ASGI server (for example: uvicorn backend.main:app --reload).
* Called by the Streamlit frontend and any API client.
"""

from fastapi import FastAPI, HTTPException

from retrieval.hybrid import hybrid_query
from retrieval.vectorstore import collection_size, list_sources
from generation.generator import generate_answer
from backend.models import AskRequest, AskResponse, SourceInfo

app = FastAPI(
    title="Curriculum RAG API",
    description="Ask questions about the college curriculum. Answers are grounded in curriculum documents only.",
    version="1.0.0",
)


# --- Endpoints ---

@app.get("/")
def root():
    """
    Return basic API information.

    Parameters:

    * None

    Returns:

    * dict: title, version, and helpful endpoint URLs.

    Steps:

    1. Build a small metadata dictionary.
    2. Return it as the root endpoint response.
    """
    return {
        "title": "Curriculum RAG API",
        "version": "1.0.0",
        "health": "/health",
        "ask_endpoint": "/ask",
    }


@app.get("/health")
def health():
    """
    Return current service health and data status.
    Parameters:
    * None
    Returns:

    * dict: API status, number of chunks, and available source files.

    Steps:

    1. Check current collection size.
    2. List source files stored in the vector database.
    3. Return all health details in one response.
    """
    return {
        "status": "ok",
        "chunks_in_db": collection_size(),
        "sources": list_sources(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Answer a user question using retrieved curriculum chunks.

    Parameters:

    * request (AskRequest): question text and retrieval options.

    Returns:

    * AskResponse: generated answer, source list, and chunk count.

    Steps:

    1. Clean and validate the incoming question.
    2. Retrieve relevant chunks using the selected retrieval mode.
    3. Generate an answer from retrieved context.
    4. Build a unique source list and return the final response.
    """
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve — filtered to branch file if provided
    chunks = hybrid_query(
        question,
        top_k=request.top_k,
        source_file=request.source_file,
        mode=request.retrieval_mode,
    )

    answer = generate_answer(question, chunks)

    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append(SourceInfo(
                source=chunk["source"],
                page=chunk["page"],
                score=chunk["score"],
            ))

    return AskResponse(
        answer=answer,
        sources=sources,
        chunks_searched=len(chunks),
    )