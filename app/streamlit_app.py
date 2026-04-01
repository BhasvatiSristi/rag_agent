"""
app/streamlit_app.py
Curriculum RAG UI — IIITDM white/blue theme with college logo.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import requests
import base64
from pathlib import Path

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Curriculum Assistant — IIITDM",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load logo as base64
def get_logo_b64():
    logo_path = Path(__file__).parent / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

LOGO_B64 = get_logo_b64()
LOGO_SRC = f"data:image/png;base64,{LOGO_B64}" if LOGO_B64 else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

[data-testid="collapsedControl"] {{ display: none !important; }}
[data-testid="stSidebar"] {{ display: none !important; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

html, body, .stApp {{
    background: #f0f4fb !important;
    color: #1a2e6e;
}}

/* ── Top navbar ── */
.navbar {{
    background: #1a3a8f;
    padding: 20px 36px;
    display: flex;
    align-items: center;
    gap: 20px;
    border-radius: 0 0 24px 24px;
    margin-bottom: 20px;
    box-shadow: 0 6px 28px rgba(26,58,143,0.22);
}}
.navbar img {{
    height: 68px;
    object-fit: contain;
    background: white;
    border-radius: 10px;
    padding: 6px 10px;
}}
.navbar-text {{
    display: flex;
    flex-direction: column;
    gap: 8px;
}}
.navbar-title-box {{
    display: inline-block;
    font-family: 'Syne', sans-serif;
    font-size: 2.0rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
}}

/* ── Branch buttons ── */
div[data-testid="stHorizontalBlock"] > div > div > div > button {{
    background: #ffffff !important;
    border: 1.5px solid #c8d8f0 !important;
    border-radius: 100px !important;
    color: #7a8fb8 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.0rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 16px 8px !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 2px 8px rgba(26,58,143,0.06) !important;
}}
div[data-testid="stHorizontalBlock"] > div > div > div > button:hover {{
    border-color: #2e7dd1 !important;
    color: #1a3a8f !important;
    background: #eef4fc !important;
    box-shadow: 0 6px 20px rgba(46,125,209,0.22) !important;
}}

/* ── Active branch pill ── */
.branch-active > div > div > div > button {{
    background: linear-gradient(135deg, #1a3a8f, #2e7dd1) !important;
    border-color: #1a3a8f !important;
    color: #ffffff !important;
    box-shadow: 0 6px 22px rgba(26,58,143,0.38) !important;
}}

/* ── Text input ── */
.stTextInput > div > div > input {{
    background: #ffffff !important;
    border: 1.5px solid #c8d8f0 !important;
    border-radius: 14px !important;
    color: #1a2e6e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 16px 20px !important;
    box-shadow: 0 2px 8px rgba(26,58,143,0.06) !important;
}}
.stTextInput > div > div > input:focus {{
    border-color: #2e7dd1 !important;
    box-shadow: 0 0 0 3px rgba(46,125,209,0.12) !important;
}}
.stTextInput > div > div > input::placeholder {{ color: #b0bdd8 !important; }}

/* ── Ask button ── */
.stButton > button {{
    background: linear-gradient(135deg, #1a3a8f, #2e7dd1) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 16px 0 !important;
    box-shadow: 0 4px 16px rgba(26,58,143,0.25) !important;
    transition: all 0.2s !important;
}}
.stButton > button:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(26,58,143,0.35) !important;
}}

/* ── Divider ── */
.my-divider {{
    border: none;
    border-top: 1px solid #dce8f8;
    margin: 10px 0 6px;
}}

/* ── Answer card ── */
.answer-wrap {{
    background: #ffffff;
    border: 1px solid #c8d8f0;
    border-radius: 18px;
    padding: 28px 30px;
    margin-top: 20px;
    box-shadow: 0 4px 24px rgba(26,58,143,0.08);
}}
.answer-label {{
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #2e7dd1;
    text-transform: uppercase;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.answer-label::before {{
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #2e7dd1;
}}
.answer-body {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 400;
    color: #2a3f6e;
    line-height: 1.8;
    white-space: pre-wrap;
}}

/* ── Source tags ── */
.sources-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}}
.src-tag {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: #2e7dd1;
    background: #eef4fc;
    border: 1px solid #c8d8f0;
    border-radius: 100px;
    padding: 6px 14px;
    text-decoration: none;
    transition: all 0.15s ease;
    cursor: pointer;
}}
.src-tag:hover {{
    background: #1a3a8f;
    color: #ffffff;
    border-color: #1a3a8f;
    box-shadow: 0 4px 12px rgba(26,58,143,0.2);
    text-decoration: none;
}}

/* ── Empty state ── */
.empty-state {{
    text-align: center;
    padding: 10px 0 4px;
    color: #b0bdd8;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.02em;
    font-weight: 300;
}}

.active-badge {{
    display: inline-block;
    background: rgba(46,125,209,0.1);
    color: #2e7dd1;
    border-radius: 100px;
    padding: 2px 10px;
    font-size: 0.7rem;
    margin-left: 6px;
}}

/* reduce gap after branch row */
div[data-testid="stHorizontalBlock"] {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}}
div[data-testid="stVerticalBlock"] > div {{
    gap: 0 !important;
}}
.block-container {{
    padding-top: 0 !important;
}}

/* ── Hero subtitle ── */
.hero {{
    text-align: center;
    padding: 6px 0 20px;
}}
.hero p {{
    color: #8a9bc4;
    font-size: 0.92rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    margin: 0;
}}

/* ── Disclaimer ── */
.disclaimer {{
    margin-top: 16px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #b0bdd8;
    line-height: 1.6;
    text-align: center;
    font-weight: 300;
    letter-spacing: 0.01em;
}}
</style>
""", unsafe_allow_html=True)


# ── Branch config ──────────────────────────────────────────────
BRANCHES = {
    "CSE":    {"label": "CSE",      "emoji": "💻", "file": "BT-CSE.pdf"},
    "CSE-AI": {"label": "CSE · AI", "emoji": "🤖", "file": "BT-CSE-AI.pdf"},
    "MECH":   {"label": "MECH",     "emoji": "⚙️", "file": "BT-ME.pdf"},
    "ECE":    {"label": "ECE",      "emoji": "📡", "file": "BT-ECE.pdf"},
    "SM":     {"label": "SM",       "emoji": "🏭", "file": "BT-SM_UPDATED.pdf"},
}

for key, default in {
    "selected_branch": "SM",
    "answer": None,
    "sources": [],
    "last_question": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Navbar with logo ───────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
    <img src="{LOGO_SRC}" alt="IIITDM Logo">
    <div class="navbar-text">
        <span class="navbar-title-box">Curriculum Assistant</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Hero subtitle ─────────────────────────────────────────────
st.markdown(
    "<div class='hero'><p>Subjects · Credits · Prerequisites · Electives — answered instantly</p></div>",
    unsafe_allow_html=True,
)

# ── Branch selector ────────────────────────────────────────────
cols = st.columns(len(BRANCHES))
for col, (key, info) in zip(cols, BRANCHES.items()):
    with col:
        is_active = st.session_state.selected_branch == key
        label = f"{info['emoji']}  {info['label']}"
        # Wrap active button in a div with class for CSS targeting
        if is_active:
            st.markdown('<div class="branch-active">', unsafe_allow_html=True)
        if st.button(label, key=f"branch_{key}", use_container_width=True):
            if not is_active:
                st.session_state.selected_branch = key
                st.session_state.answer = None
                st.session_state.sources = []
                st.session_state.last_question = ""
                st.rerun()
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)

active_key = st.session_state.selected_branch
active_info = BRANCHES[active_key]

st.markdown(
    "<div class='empty-state'>— select a branch · type your question —</div>",
    unsafe_allow_html=True,
)

st.markdown("<hr class='my-divider'>", unsafe_allow_html=True)


# ── Question input ─────────────────────────────────────────────
question = st.text_input(
    label="q",
    placeholder=f"Ask about {active_info['label']} — subjects, credits, prerequisites...",
    label_visibility="collapsed",
    key="q_input",
)

ask_clicked = st.button("Ask →", use_container_width=True)


# ── Query ──────────────────────────────────────────────────────
if ask_clicked and question.strip():
    with st.spinner("Thinking..."):
        try:
            resp = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": question.strip(),
                    "top_k": 5,
                    "source_file": active_info["file"],
                },
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.answer = data["answer"]
                st.session_state.sources = data.get("sources", [])
                st.session_state.last_question = question.strip()
            else:
                st.session_state.answer = f"API error {resp.status_code}: {resp.text}"
                st.session_state.sources = []
        except requests.exceptions.ConnectionError:
            st.session_state.answer = "⚠️ API offline.\n\nRun: uvicorn api.main:app --reload --port 8000"
            st.session_state.sources = []
        except Exception as e:
            st.session_state.answer = f"Error: {e}"
            st.session_state.sources = []


# ── Answer ─────────────────────────────────────────────────────
if st.session_state.answer:
    sources_html = "".join(
        f'<span class="src-tag">'
        f'📄 {s["source"]} &middot; p{s["page"]} &middot; {s["score"]:.2f}</span>'
        for s in st.session_state.sources
    )
    st.markdown(f"""
    <div class="answer-wrap">
        <div class="answer-label">
            {active_info['emoji']} {active_info['label']}
            <span class="active-badge">answer</span>
        </div>
        <div class="answer-body">{st.session_state.answer}</div>
    </div>
    <div class="sources-row">{sources_html}</div>
    <div class="disclaimer">⚠️ Answers may not be accurate. Please verify with the official documents linked above.</div>
    """, unsafe_allow_html=True)


# ── Disclaimer ────────────────────────────────────────────────
st.markdown(
    "<div class='disclaimer'>"
    "⚠️ Please verify the actual documents mentioned below for more accurate results. "
    "These answers are based on the <strong>B.Tech Curriculum 2020</strong>."
    "</div>",
    unsafe_allow_html=True,
)