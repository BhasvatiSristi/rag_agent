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
- `api/main.py` exposes the FastAPI endpoints.
- `app/streamlit_app.py` is the UI.

## Before vs After
### Before
- Extra helper layers and wrappers.
- Chroma debug output mixed into normal query paths.
- Stale UI branch reference for Smart Manufacturing.

### After
- Dead helpers removed.
- Retrieval and chunking logic is flatter and easier to read.
- Constants are named where they are used.
- The UI points to the correct curriculum file.

## Deploy (Render)
1. Push this repo to GitHub.
2. In Render, create a Blueprint and select this repo.
3. Render will read `render.yaml` and create two services:
	- `rag-agent-api`
	- `rag-agent-ui`
4. In Render dashboard, set env vars:
	- `HF_API_TOKEN` on `rag-agent-api`
	- `LLM_PROVIDER=hf` on `rag-agent-api`
	- `HF_LLM_MODEL=deepseek-ai/DeepSeek-R1` on `rag-agent-api`
	- `API_URL` on `rag-agent-ui` to your API public URL (for example `https://rag-agent-api.onrender.com`).
5. Open API shell once and run ingestion:
	- `python ingest_pipeline.py --data-dir ./data/raw`

After ingestion completes, open the Streamlit UI URL and start asking questions.