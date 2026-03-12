# 🎓 Curriculum RAG — B.Tech Smart Manufacturing

A lightweight RAG (Retrieval-Augmented Generation) system to answer questions about the IIITDM Kancheepuram B.Tech Smart Manufacturing curriculum.

Answers are grounded **strictly** in the curriculum documents. No hallucinations.

---

## 🏗️ Architecture

```
PDF → Extract Tables + Text → Chunk (~700 tokens) → Embed (BGE-small) → ChromaDB
                                                                              ↓
User Question → Embed → Retrieve Top-K Chunks → Ollama LLM → Answer
```

---

## ⚡ Quick Start

### 1. Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

```bash
# Install and pull a model
ollama pull llama3       # recommended (~4GB)
# or
ollama pull mistral      # lighter alternative (~4GB)
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (default settings work out of the box)
```

### 4. Add your PDF(s)

```bash
# Place curriculum PDFs in:
data/raw/
```

### 5. Ingest documents (run once)

```bash
python ingest_pipeline.py
```

### 6. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 7. Test from terminal

```bash
# Interactive mode
python test_query.py

# Single question
python test_query.py --question "What subjects are in semester 3?"

# Run all sample questions
python test_query.py --sample

# With verbose chunk output
python test_query.py --question "What are the electives?" --verbose
```

### 8. Optional: Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 API Reference

### `POST /ask`

**Request:**
```json
{
  "question": "What subjects are in semester 3?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "Semester 3 includes the following subjects: ...",
  "sources": [
    { "source": "BT-SM_UPDATED.pdf", "page": 2, "score": 0.91 }
  ],
  "chunks_searched": 5
}
```

### `GET /health`

```json
{ "status": "ok", "chunks_in_db": 142, "ready": true }
```

Interactive docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
curriculum-rag/
├── data/raw/               # Drop PDF files here
├── ingestion/
│   ├── loader.py           # PDF loading + table extraction
│   └── chunker.py          # Token-aware chunking
├── retrieval/
│   ├── embedder.py         # SentenceTransformer (BAAI/bge-small-en)
│   └── vectorstore.py      # ChromaDB operations
├── generation/
│   └── llm.py              # Ollama LLM + strict prompt
├── api/
│   └── main.py             # FastAPI endpoints
├── app/
│   └── streamlit_app.py    # Optional UI
├── config/
│   └── settings.py         # Central config from .env
├── ingest_pipeline.py      # Run once to ingest PDFs
├── test_query.py           # Terminal testing
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3` | Ollama model to use |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en` | HuggingFace embedding model |
| `CHUNK_SIZE` | `700` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap tokens between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |

---

## 🔒 No-Hallucination Policy

The LLM is instructed to:
1. Answer **only** from the provided context
2. If context doesn't support the answer, respond with:
   > *"I could not find supporting information in the curriculum documents."*
