"""
api/main.py
FastAPI app — POST /ask with optional branch (source file) filtering.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import threading

from retrieval.hybrid import hybrid_query
from retrieval.vectorstore import collection_size, list_sources
from generation.generator import generate_answer
from ingest_pipeline import run_ingestion
from api.models import AskRequest, AskResponse, SourceInfo


AUTO_INGEST_ON_EMPTY = os.getenv("AUTO_INGEST_ON_EMPTY", "true").strip().lower() in {"1", "true", "yes", "on"}
INGEST_DATA_DIR = os.getenv("INGEST_DATA_DIR", "./data/raw")
_AUTO_INGEST_ATTEMPTED = False
_AUTO_INGEST_LOCK = threading.Lock()

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


def _maybe_auto_ingest() -> None:
    """Attempt one-time ingestion if vector DB is empty and auto-ingest is enabled."""
    global _AUTO_INGEST_ATTEMPTED

    if not AUTO_INGEST_ON_EMPTY or _AUTO_INGEST_ATTEMPTED:
        return

    with _AUTO_INGEST_LOCK:
        if _AUTO_INGEST_ATTEMPTED:
            return
        _AUTO_INGEST_ATTEMPTED = True

        if collection_size() > 0:
            return

        try:
            run_ingestion(INGEST_DATA_DIR)
        except Exception as e:
            # Keep API available even if ingestion fails (e.g., missing keys/files).
            print(f"[startup] Auto-ingestion skipped/failed: {e}")


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
        _maybe_auto_ingest()

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