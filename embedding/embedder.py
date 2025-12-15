from langchain_huggingface import HuggingFaceEmbeddings

# For E5 models, you need to add instruction prefixes
def get_e5_embeddings():
    model_name = "intfloat/multilingual-e5-large"  # or "multilingual-e5-base" for faster
    model_kwargs = {'device': 'cpu'}  # Use 'cpu' if no GPU
    encode_kwargs = {'normalize_embeddings': True}  # Important for cosine similarity

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# E5 models work best with instruction prefixes
def add_e5_prefix_to_docs(docs_split):
    """Add 'passage: ' prefix for E5 models"""
    for doc in docs_split:
        doc.page_content = f"passage: {doc.page_content}"
    return docs_split