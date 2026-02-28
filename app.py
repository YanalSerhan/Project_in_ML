import streamlit as st
import os
import json
import uuid
from pathlib import Path
from datetime import datetime

from RAG.rag import RAG
from queryProcess.enhancer import QueryEnhancer
from sql_retrieval.sql_converter import SQL_converter
from embedding.embedder import E5Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ─────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────
#  HISTORY HELPERS
# ─────────────────────────────────────────────────────────
def history_path(session_id: str) -> Path:
    return HISTORY_DIR / f"{session_id}.json"

def save_session(session_id: str, messages: list, conv_state: dict):
    data = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "conv_state": conv_state,
    }
    with open(history_path(session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_session(session_id: str) -> dict | None:
    p = history_path(session_id)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None

def list_sessions() -> list[dict]:
    """Return sessions sorted newest-first with a preview label."""
    sessions = []
    for p in sorted(HISTORY_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            first_user = next((m["content"] for m in d.get("messages", []) if m["role"] == "user"), "Empty session")
            sessions.append({
                "id": d["session_id"],
                "label": first_user[:42] + ("…" if len(first_user) > 42 else ""),
                "timestamp": d.get("timestamp", ""),
                "msg_count": len(d.get("messages", [])),
            })
        except Exception:
            pass
    return sessions

def delete_session(session_id: str):
    p = history_path(session_id)
    if p.exists():
        p.unlink()

# ─────────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #13161d;
    --border:    #1e2330;
    --accent:    #4f7cff;
    --accent2:   #8b5cf6;
    --text:      #e8eaf0;
    --muted:     #6b7280;
    --user-bg:   #1a1f2e;
    --bot-bg:    #131820;
    --danger:    #ef4444;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
header { background-color: transparent !important; }
.stDeployButton { display: none; }

.main .block-container {
    max-width: 780px;
    padding: 2rem 1.5rem 6rem;
    margin: 0 auto;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 30%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem !important;
}
.app-header p { color: var(--muted) !important; font-size: 0.92rem; font-weight: 300; margin: 0; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(79,124,255,0.08);
    border: 1px solid rgba(79,124,255,0.25);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.73rem; color: var(--accent);
    font-family: 'DM Mono', monospace; margin-top: 0.9rem;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #22c55e; box-shadow: 0 0 6px #22c55e;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* ── Chat bubbles ── */
[data-testid="stChatMessage"][data-role="user"] {
    background: var(--user-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px 16px 4px 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 1rem !important;
}
[data-testid="stChatMessage"][data-role="assistant"] {
    background: var(--bot-bg) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 1rem !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79,124,255,0.12) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── ALL buttons — unified style, no conflicts ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    padding: 0.4rem 0.75rem !important;
    width: 100% !important;
    transition: border-color 0.18s, color 0.18s, background 0.18s !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(79,124,255,0.06) !important;
}
/* Danger variant — applied via container class */
.danger-btn .stButton > button:hover {
    border-color: var(--danger) !important;
    color: var(--danger) !important;
    background: rgba(239,68,68,0.06) !important;
}

/* ── Stat / context cards ── */
.ctx-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}
.ctx-label {
    font-size: 0.68rem; color: var(--muted);
    font-family: 'DM Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}
.ctx-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 4px; }
.ctx-tag {
    background: rgba(79,124,255,0.1);
    border: 1px solid rgba(79,124,255,0.22);
    border-radius: 12px; padding: 2px 9px;
    font-size: 0.72rem; color: var(--accent);
    font-family: 'DM Mono', monospace;
}
.ctx-tag.lec {
    background: rgba(139,92,246,0.1);
    border-color: rgba(139,92,246,0.25);
    color: var(--accent2);
}
.ctx-empty { font-size: 0.78rem; color: #374151; font-style: italic; }

/* ── Past session list ── */
.sess-section-label {
    font-size: 0.62rem;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    text-transform: uppercase;
    letter-spacing: .1em;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sess-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}
.sess-empty {
    font-size: 0.76rem;
    color: #374151;
    font-style: italic;
    text-align: center;
    padding: 1rem 0.5rem;
    border: 1px dashed var(--border);
    border-radius: 8px;
}

/* ── Session Load Button (Left Side) ── */
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(n+2) div[data-testid="column"]:nth-child(1) .stButton > button {
    background: linear-gradient(135deg, rgba(19,22,29,0.95) 0%, rgba(13,15,20,0.95) 100%) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid transparent !important;
    border-radius: 10px !important;
    padding: 0.65rem 0.85rem !important;
    width: 100% !important;
    height: auto !important;
    min-height: 3.2rem !important;
    text-align: left !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
    white-space: normal !important;
    word-break: break-word !important;
    transition: border-color 0.18s, background 0.18s, transform 0.12s !important;
    cursor: pointer !important;
    margin-bottom: 0.3rem !important;
}

div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(n+2) div[data-testid="column"]:nth-child(1) .stButton > button:hover {
    border-left-color: var(--accent) !important;
    border-color: rgba(79,124,255,0.4) !important;
    background: linear-gradient(135deg, rgba(79,124,255,0.08) 0%, rgba(13,15,20,0.95) 100%) !important;
    transform: translateX(2px) !important;
    color: var(--text) !important;
}

/* ── Session Delete Button (Right Side - "X") ── */
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(n+2) div[data-testid="column"]:nth-child(2) .stButton > button {
    background: linear-gradient(135deg, rgba(19,22,29,0.95) 0%, rgba(13,15,20,0.95) 100%) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: 0.9rem !important;
    width: 100% !important;
    height: auto !important;
    min-height: 3.2rem !important; /* Matches exactly with load button */
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    margin-bottom: 0.3rem !important;
}

div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"]:nth-of-type(n+2) div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    border-color: rgba(239, 68, 68, 0.4) !important;
    color: var(--danger) !important;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(13, 15, 20, 0.95) 100%) !important;
    transform: scale(0.96) !important; /* Subtle press effect */
}

/* ── Empty state ── */
.empty-state { text-align: center; padding: 3rem 1rem; color: var(--muted); }
.empty-state .icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.4; }
.empty-state p { font-size: 0.9rem; line-height: 1.8; }

/* ── Suggestion chip buttons ── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    border-radius: 20px !important;
    padding: 5px 14px !important;
    font-size: 0.78rem !important;
    font-family: 'DM Sans', sans-serif !important;
    white-space: nowrap !important;
    height: auto !important;
    width: auto !important;
}

/* ── Force equal-height sidebar action buttons ── */
div[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] .stButton > button {
    height: 2.2rem !important;
    line-height: 1 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    border-radius: 8px !important;
}

/* ── Error ── */
.stAlert {
    background: rgba(239,68,68,0.06) !important;
    border: 1px solid rgba(239,68,68,0.3) !important;
    border-radius: 10px !important; color: #fca5a5 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  RESOURCE LOADING
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embeddings_class = E5Embeddings()
    vectorstore      = Chroma(persist_directory="chroma_db",       embedding_function=embeddings)
    classification_db= Chroma(persist_directory="classification_db", embedding_function=embeddings_class)
    db_schemas       = Chroma(persist_directory="db_schemas",      embedding_function=embeddings)
    model_id = "deepseek-ai/deepseek-v3.2"
    enhancer = QueryEnhancer(model_id)
    sql_tool = SQL_converter(model_id)
    return vectorstore, classification_db, enhancer, sql_tool, db_schemas

v_store, c_db, enhancer, sql_tool, schemas = load_resources()


# ─────────────────────────────────────────────────────────
#  SESSION STATE BOOTSTRAP
# ─────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_state" not in st.session_state:
    st.session_state.conv_state = {"course": [], "lecturer": []}


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-family:DM Serif Display,serif;font-size:1.3rem;"
        "font-weight:400;margin:0.5rem 0 0.2rem;color:#e8eaf0;'>Academic Assistant</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.68rem;color:#374151;font-family:DM Mono,monospace;"
        "margin:0 0 0.8rem;'>Main CS · RAG System v2</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Context tracker ──────────────────────────────────
    courses   = st.session_state.conv_state.get("course", [])
    lecturers = st.session_state.conv_state.get("lecturer", [])

    course_tags   = "".join(f'<span class="ctx-tag">{c}</span>'   for c in courses)   or '<span class="ctx-empty">none yet</span>'
    lecturer_tags = "".join(f'<span class="ctx-tag lec">{l}</span>' for l in lecturers) or '<span class="ctx-empty">none yet</span>'

    st.markdown(f"""
    <div class="ctx-card">
        <div class="ctx-label">📚 Courses in context</div>
        <div class="ctx-tags">{course_tags}</div>
    </div>
    <div class="ctx-card">
        <div class="ctx-label">🧑‍🏫 Lecturers in context</div>
        <div class="ctx-tags">{lecturer_tags}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

# ── New / Clear conversation ──────────────────────────
    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("＋ New chat", use_container_width=True):
            # Save current session first if it has messages
            if st.session_state.messages:
                save_session(
                    st.session_state.session_id,
                    st.session_state.messages,
                    st.session_state.conv_state,
                )
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.session_state.conv_state = {"course": [], "lecturer": []}
            st.rerun()
    with col_clear:
        if st.button("🗑 Clear", use_container_width=True):
            delete_session(st.session_state.session_id)
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages   = []
            st.session_state.conv_state = {"course": [], "lecturer": []}
            st.rerun()

    st.divider()

# ── Past sessions ─────────────────────────────────────
    st.markdown(
        "<div class='sess-section-label'>Past sessions</div>",
        unsafe_allow_html=True,
    )

    all_sessions = list_sessions()
    past_sessions = [s for s in all_sessions if s["id"] != st.session_state.session_id]

    if not past_sessions:
        st.markdown("<div class='sess-empty'>No saved sessions yet.</div>", unsafe_allow_html=True)
    else:
        for sess in past_sessions[:10]:
            ts_raw = sess["timestamp"]
            try:
                dt = datetime.fromisoformat(ts_raw)
                today = datetime.now().date()
                if dt.date() == today:
                    ts = f"Today {dt.strftime('%H:%M')}"
                else:
                    ts = dt.strftime("%b %d")
            except Exception:
                ts = ts_raw[:10] if ts_raw else ""

            msg_count = sess["msg_count"]
            label = f"💬  {sess['label']}\n\n{ts}  ·  {msg_count} msgs"

            # Use an 85/15 ratio so the X button is slightly wider and better proportioned
            col_load, col_del = st.columns([85, 15])
            
            with col_load:
                if st.button(label, key=f"sess_{sess['id']}", use_container_width=True):
                    if st.session_state.messages:
                        save_session(
                            st.session_state.session_id,
                            st.session_state.messages,
                            st.session_state.conv_state,
                        )
                    loaded = load_session(sess["id"])
                    if loaded:
                        st.session_state.session_id = loaded["session_id"]
                        st.session_state.messages   = loaded["messages"]
                        st.session_state.conv_state = loaded["conv_state"]
                        st.rerun()

            with col_del:
                if st.button("✕", key=f"del_{sess['id']}", help="Delete session", use_container_width=True):
                    delete_session(sess["id"])
                    st.rerun()


# ─────────────────────────────────────────────────────────
#  MAIN AREA — HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>Academic Assistant</h1>
    <p>Course reviews · Grade data · Prerequisites · Lecturer info</p>
    <div class="status-badge">
        <span class="status-dot"></span>
        deepseek-v3.2 · multilingual-e5-large
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  EMPTY STATE
# ─────────────────────────────────────────────────────────
SUGGESTIONS = [
    "מה הציון הממוצע בקורס תכנון וניתוח אלגוריתמים ?",
    "חוות דעת על קורס מבני נתונים?",
    "מה הדרישות קדם לקורס מבוא ללמידה ממוכנת?",
]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ONLY show the empty state if there are no messages AND no prompt is pending
if not st.session_state.messages and not st.session_state.pending_prompt:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🎓</div>
        <p>Ask me anything about courses, lecturers,<br>grades, or prerequisites.</p>
    </div>
    """, unsafe_allow_html=True)

    # Real clickable suggestion chips in a 2×2 grid
    row1_cols = st.columns(2)
    row2_cols = st.columns(2)
    chip_slots = [row1_cols[0], row1_cols[1], row2_cols[0], row2_cols[1]]
    for col, suggestion in zip(chip_slots, SUGGESTIONS):
        with col:
            if st.button(suggestion, key=f"chip_{suggestion}", use_container_width=True):
                st.session_state.pending_prompt = suggestion
                st.rerun()


# ─────────────────────────────────────────────────────────
#  CHAT HISTORY DISPLAY
# ─────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ─────────────────────────────────────────────────────────
#  INPUT & RAG RESPONSE
# ─────────────────────────────────────────────────────────
# Resolve prompt — either from chip click or typed input
prompt = st.chat_input("שאל אותי משהו על הקורסים או המרצים…")
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving & generating…"):
            try:
                response = RAG(
                    query=prompt,
                    query_enhancer=enhancer,
                    vectorstore=v_store,
                    classification_db=c_db,
                    sql_converter=sql_tool,
                    conv_state=st.session_state.conv_state,
                    db_schema=schemas,
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

                # Auto-save after every assistant turn
                save_session(
                    st.session_state.session_id,
                    st.session_state.messages,
                    st.session_state.conv_state,
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
