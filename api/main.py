"""
api/main.py
FastAPI app — POST /ask with optional branch (source file) filtering.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from retrieval.hybrid import hybrid_query
from retrieval.vectorstore import collection_size, list_sources
from generation.generator import generate_answer
from config.settings import TOP_K, RETRIEVAL_MODE

app = FastAPI(
    title="Curriculum RAG API",
    description="Ask questions about the college curriculum. Answers are grounded in curriculum documents only.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---

class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K
    source_file: Optional[str] = None   # e.g. "BT-ME.pdf" — filters to one branch
    retrieval_mode: str = RETRIEVAL_MODE


class SourceInfo(BaseModel):
    source: str
    page: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    chunks_searched: int


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "title": "Curriculum RAG API",
        "version": "1.0.0",
        "docs": "http://localhost:8000/docs",
        "health": "http://localhost:8000/health",
        "ask_endpoint": "POST http://localhost:8000/ask",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks_in_db": collection_size(),
        "sources": list_sources(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection_size() == 0:
        raise HTTPException(status_code=503, detail="Vector DB is empty. Run ingest_pipeline.py first.")

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