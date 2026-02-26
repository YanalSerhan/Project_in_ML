import streamlit as st
import os
import json
import re
from pathlib import Path

# --- 1. IMPORTS & SYSTEM SETUP ---
from RAG.rag import RAG 
from queryProcess.enhancer import QueryEnhancer
from sql_retrieval.sql_converter import SQL_converter
from embedding.embedder import E5Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 2. PROFESSIONAL UI STYLING ---
st.set_page_config(page_title="Main CS: Academic Assistant", layout="centered", page_icon="🎓")

# Custom CSS for Professional Look and Hebrew RTL support
st.markdown("""
    <style>
    /* Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Assistant:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', 'Assistant', sans-serif;
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1E3A8A; /* University Blue */
        text-align: center;
        margin-bottom: 0px;
    }
    
    /* Sub-header */
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* RTL Support for Hebrew */
    .rtl-text {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Assistant', sans-serif;
        line-height: 1.6;
    }
    
    /* Chat Bubble Tweaks */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def is_hebrew(text):
    """Detects if a string contains Hebrew characters for RTL alignment."""
    return bool(re.search(r'[\u0590-\u05FF]', text))

@st.cache_resource
def load_resources():
    """Load models and databases once and cache them for speed."""
    # Setup exactly as in your main.py / notebooks
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embeddings_class = E5Embeddings()
    
    # Load Vector Stores
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    classification_db = Chroma(persist_directory="classification_db", embedding_function=embeddings_class)
    db_schemas = Chroma(persist_directory="db_schemas", embedding_function=embeddings)
    
    # Initialize logic tools with DeepSeek
    model_id = "deepseek-ai/deepseek-v3.2"
    enhancer = QueryEnhancer(model_id)
    sql_tool = SQL_converter(model_id)
    
    return vectorstore, classification_db, enhancer, sql_tool, db_schemas

# Load the heavy resources
v_store, c_db, enhancer, sql_tool, schemas = load_resources()

# --- 4. UI HEADER ---
st.markdown('<div class="main-header">🎓 Main CS: Academic Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered insights for University of Haifa CS Students</div>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2997/2997313.png", width=100)
    st.header("Control Panel")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conv_state = {"course": [], "lecturer": []}
        st.rerun()
    st.divider()
    st.info("System: RAG over SQL and Semantic DB.")

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_state" not in st.session_state:
    st.session_state.conv_state = {"course": [], "lecturer": []}

# --- 5. CHAT LOOP ---
# Display messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if is_hebrew(message["content"]):
            st.markdown(f'<div class="rtl-text">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# User input field
if prompt := st.chat_input("תשאל אותי על מרצים, קורסים וקדמים"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI Response
    with st.chat_message("assistant"):
        # נשתמש ב-spinner פשוט שנעלם בסיום העיבוד
        with st.spinner("מעבד את השאילתה שלך..."):
            try:
                # Run the RAG pipeline
                response = RAG(
                    query=prompt,
                    query_enhancer=enhancer,
                    vectorstore=v_store,
                    classification_db=c_db,
                    sql_converter=sql_tool,
                    conv_state=st.session_state.conv_state,
                    db_schema=schemas
                )
            except Exception as e:
                st.error(f"אירעה שגיאה: {e}")
                response = "מצטער, אירעה שגיאה בעיבוד הבקשה."

        # הצגת התשובה מיד כשהיא מוכנה - ללא צורך בלחיצה
        if response:
            if is_hebrew(response):
                st.markdown(f'<div class="rtl-text">{response}</div>', unsafe_allow_html=True)
            else:
                st.markdown(response)
            
            # שמירה בהיסטוריה
            st.session_state.messages.append({"role": "assistant", "content": response})