"""
Purpose:

* Provides a Streamlit user interface to ask curriculum questions.

Inputs:

* User-selected branch and question text from UI widgets.

Outputs:

* Visual answer display and source list fetched from FastAPI.

Used in:

* Run as frontend app that calls the API ask endpoint.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Curriculum Assistant",
    page_icon="🎓",
    layout="wide",
)


# ── Branch config ──────────────────────────────────────────────
BRANCHES = {
    "CSE":    {"file": "BT-CSE.pdf"},
    "CSE-AI": {"file": "BT-CSE-AI.pdf"},
    "MECH":   {"file": "BT-ME.pdf"},
    "ECE":    {"file": "BT-ECE.pdf"},
    "SM":     {"file": "BT-SM.pdf"},
}

# Initialize session state
if "selected_branch" not in st.session_state:
    st.session_state.selected_branch = "CSE"
if "answer" not in st.session_state:
    st.session_state.answer = None
if "sources" not in st.session_state:
    st.session_state.sources = []


# ── Page title ─────────────────────────────────────────────────
st.title("🎓 Curriculum Assistant")
st.write("Ask questions about the B.Tech curriculum")

# ── Branch selector ────────────────────────────────────────────
st.write("**Select a branch:**")
selected = st.radio("", list(BRANCHES.keys()), horizontal=True, key="branch_select")
st.session_state.selected_branch = selected

# ── Question input ─────────────────────────────────────────────
question = st.text_input("Ask your question:")

# ── Query ──────────────────────────────────────────────────────
if question.strip():
    with st.spinner("Searching..."):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question,
                    "top_k": 5,
                    "source_file": BRANCHES[st.session_state.selected_branch]["file"],
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.answer = data["answer"]
                st.session_state.sources = data.get("sources", [])
            else:
                st.error(f"API error: {resp.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ API offline. Start it with: `uvicorn backend.main:app --reload`")
        except Exception as e:
            st.error(f"Error: {e}")


# ── Answer display ─────────────────────────────────────────────
if st.session_state.answer:
    st.write("---")
    st.write("**Answer:**")
    st.write(st.session_state.answer)

    if st.session_state.sources:
        st.write("**Sources:**")
        for src in st.session_state.sources:
            st.write(f"- {src['source']} (page {src['page']}, score: {src['score']})")

st.write("---")
st.caption("⚠️ Verify answers with the official documents.")