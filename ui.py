import streamlit as st
import os
import json
from pathlib import Path

# Updated imports to specifically target the RAG function and classes
from RAG.rag import RAG 
#from generation.answerGenerator import AnswerGenerator
from queryProcess.enhancer import QueryEnhancer
from sql_retrieval.sql_converter import SQL_converter
from embedding.embedder import E5Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. INITIALIZATION & CACHING ---
@st.cache_resource
def load_resources():
    """Load heavy models and databases once and cache them."""
    # Using the same embedding setup as main.py
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    embeddings_class = E5Embeddings()
    
    # Load Vector Stores
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    classification_db = Chroma(persist_directory="classification_db", embedding_function=embeddings_class)
    
    # Load SQL Schemas from Chroma (as seen in main.py)
    db_schemas = Chroma(persist_directory="db_schemas", embedding_function=embeddings)
    
    # Initialize tools with the DeepSeek model used in main.py
    model_id = "deepseek-ai/deepseek-v3.2"
    enhancer = QueryEnhancer(model_id)
    sql_tool = SQL_converter(model_id)
    #answerGen = AnswerGenerator()
    
    return vectorstore, classification_db, enhancer, sql_tool, db_schemas

# Load resources
v_store, c_db, enhancer, sql_tool, schemas = load_resources()

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="Main CS: Academic Assistant", layout="centered")

st.title("🎓 Main CS: Academic Assistant")
st.markdown("Ask about student reviews, course grades, or prerequisites.")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conv_state = {"course": [], "lecturer": []}

# Initialize session state for chat and conversation memory
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conv_state" not in st.session_state:
    # This state is passed to RAG to maintain multi-turn context
    st.session_state.conv_state = {"course": [], "lecturer": []}

# --- 3. CHAT INTERFACE ---
# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("שאל אותי משהו על הקורסים או המרצים!"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # FIXED: Calling the RAG function with correct arguments
                response = RAG(
                    query=prompt,
                    query_enhancer=enhancer,
                    vectorstore=v_store,
                    classification_db=c_db,
                    sql_converter=sql_tool,
                    conv_state=st.session_state.conv_state,
                    db_schema=schemas
                )
                st.markdown(response)
                # Store assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")