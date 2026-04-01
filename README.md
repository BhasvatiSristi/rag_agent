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
- Embeddings: BAAI/bge-small-en
- Vector DB: ChromaDB
- LLM: Groq API (llama-3.1-8b-instant)
- UI: Streamlit

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