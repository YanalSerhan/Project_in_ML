from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# PASTE YOUR NEW KEY INSIDE THE QUOTES BELOW
api_key = "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" 

# Initialize the Embedder
embedder = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    api_key=api_key
)

print(" Embedder initialized successfully.")