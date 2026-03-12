---
title: Curriculum RAG
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.36.0"
app_file: app/hf_app.py
pinned: false
---

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
- LLM: Groq (llama3-8b-8192)
- UI: Streamlit