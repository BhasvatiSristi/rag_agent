# 🎓 Curriculum Assistant — IIITDM Kancheepuram

A RAG-powered assistant for the B.Tech curriculum at IIITDM Kancheepuram.

Ask questions about subjects, credits, prerequisites, and electives across all branches — CSE, CSE-AI, MECH, ECE, and Smart Manufacturing.

Answers are grounded strictly in the curriculum documents (B.Tech 2020).

## Branches Supported
- 💻 CSE
- 🤖 CSE · AI
- ⚙️ MECH
- 📡 ECE
- 🏭 Smart Manufacturing

## Stack
- Embeddings: Cohere API (embed-english-v3.0)
- Vector DB: ChromaDB
- LLM: Groq API (llama-3.1-8b-instant)
- UI: Streamlit

## How It Works
1. PDFs are loaded from `data/raw/`.
2. Each page is chunked into smaller text blocks.
3. Chunks are stored in ChromaDB and indexed with BM25.
4. A question is answered with hybrid retrieval: dense search + BM25.
5. The retrieved chunks are sent to the LLM to generate a grounded answer.

## Simplified Structure
- `ingest_pipeline.py` handles the one-time ingestion flow.
- `ingestion/loader.py` reads PDFs into page text.
- `ingestion/chunker.py` turns pages into overlapping chunks.
- `retrieval/vectorstore.py` stores and queries dense chunks.
- `retrieval/bm25.py` stores and queries keyword matches.
- `retrieval/hybrid.py` merges both retrieval modes.
- `generation/generator.py` builds the final prompt and calls the LLM.
- `backend/main.py` exposes the FastAPI endpoints.
- `frontend/streamlit_app.py` is the UI.
