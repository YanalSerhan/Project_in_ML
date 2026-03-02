from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain_core.documents import Document

def rerank_documents(query: str, documents: list[Document], model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2") -> list[Document]:
    """
    Reranks a list of documents based on their relevance to a given query
    using an NVIDIA reranking model.

    Args:
        query: The user's query.
        documents: A list of Document objects to be reranked.
        model_name: The name of the NVIDIA reranking model to use.

    Returns:
        A list of Document objects, reranked by relevance.
    """
    print("Entered Reranker File")
    if not documents:
        return []
    reranker = NVIDIARerank(model=model_name, top_n=len(documents))
    reranked_docs = reranker.compress_documents(documents, query)
    return reranked_docs
