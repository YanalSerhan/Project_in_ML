from langchain_chroma import Chroma

def create_vector_store(docs_split, embeddings, persist_directory="./chroma_db"):
    """Create and persist vector store"""
    vectorstore = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore