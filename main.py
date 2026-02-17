from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma

from loader.load_reviews import load_reviews
from loader.load_ids import load_ids
from embedding.embedder import get_e5_embeddings, E5Embeddings
from chunking.chunker import chunk_docs
from queryProcess.enhancer import QueryEnhancer
from knowledgeBase.slot_filler import SlotFiller
from knowledgeBase.conversation_state import ConversationState
from query_type.queryType import queryType
from sql_retrieval.sql_converter import SQL_converter
from RAG.rag import RAG

from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

print("Starting RAG system...")

load_dotenv()
app = FastAPI()


# Allow frontend (React / HTML) to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Global Initialization ---------

docs = load_reviews()
courses, lecturers = load_ids()

#docs_split = chunk_docs(docs)
#docs_split = add_e5_prefix_to_docs(docs_split)
embeddings = get_e5_embeddings()
embeddings_ = E5Embeddings()

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

db_schemas = Chroma(
    persist_directory="db_schemas",
    embedding_function=embeddings
)

classification_vectorstore = Chroma(
    persist_directory="classification_db",
    embedding_function=embeddings_
)

query_enhancer = QueryEnhancer("deepseek-ai/deepseek-v3.1")
#query_typer = queryType("deepseek-ai/deepseek-v3.1")
sql_converter = SQL_converter("deepseek-ai/deepseek-v3.1")
KB = ConversationState()
#slot_filler = SlotFiller()

# --------- API Schemas ---------

class QueryRequest(BaseModel):
    query: str


class RetrievedChunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = {}
    score: float | None = None


class RAGResponse(BaseModel):
    answer: str


# --------- Endpoints ---------

@app.post("/rag", response_model=RAGResponse)
async def rag_endpoint(req: QueryRequest):
    answer = RAG(
        req.query, query_enhancer, vectorstore, classification_vectorstore, sql_converter, conv_state=KB,
        db_schema=db_schemas
    )

    return RAGResponse(
        answer=answer
    )


@app.post("/enhance")
async def enhance_endpoint(req: QueryRequest):
    enhanced = query_enhancer.enhance(req.query)
    return {"enhanced_query": enhanced}


@app.post("/retrieve")
async def retrieve_endpoint(req: QueryRequest):
    chunks = vectorstore.similarity_search_with_score(req.query, k=5)

    formatted = [
        {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
        }
        for doc, score in chunks
    ]
    return {"chunks": formatted}


@app.get("/")
def root():
    return {"message": "RAG system is running!"}


# To run the application, use the command: # uvicorn main:app --reload