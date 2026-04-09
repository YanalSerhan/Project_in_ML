from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma  # or use FAISS, Pinecone, etc.
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter # Updated import
import json

class E5Embeddings(Embeddings):
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        texts = [f"passage: {t}" for t in texts]
        return self.model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True
        )[0].tolist()

def create_vector_store(docs_split, embeddings, persist_directory="./chroma_db"):
    vectorstore = Chroma.from_documents(
        documents=docs_split,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

# E5 models work best with instruction prefixes
def add_e5_prefix_to_docs(docs_split):
    """Add 'passage: ' prefix for E5 models"""
    for doc in docs_split:
        doc.page_content = f"passage: {doc.page_content}"
    return docs_split


with open("data/cleaned_reviews.json", encoding="utf-8") as f:
    data = json.load(f)
    f.close()

docs = []

for course in data:
    for i, r in enumerate(course["reviews"]):
        docs.append(
            Document(
                page_content=r["content"],
                metadata={
                    "course_id": course["course_id"],
                    "course_name": course["course_name"],
                    "lecturer": course["lecturer"],
                    "rank": r["rank"],
                    "date": r["time"],
                    "review_idx": i
                }
            )
        )

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap= 50,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

## Some nice custom preprocessing
# documents[0].page_content = documents[0].page_content.replace(". .", "")
docs_split = text_splitter.split_documents(docs)

# assign chunk id
for i, doc in enumerate(docs_split):
    doc.metadata["id"] = i

docs_split = add_e5_prefix_to_docs(docs_split)
embeddings = E5Embeddings()
vectorstore = create_vector_store(docs_split, embeddings)
print("Vector store created at ./chroma_db")


# load schema tables 
with open("data/tables.json", encoding="utf-8") as f:
    schema_documents = json.load(f)
    f.close()

schema_docs_for_vectorstore = []
for schema in schema_documents:
    combined_text = schema["content"] + " " + " ".join(schema["example_queries"]) + " " + " ".join(schema["negative_scope"])

    # Prepare metadata for Chroma (ensure simple types or JSON strings)
    #schema["columns"] = json.dumps(schema["columns"]) # Convert list of dicts to string

    schema_docs_for_vectorstore.append(
        Document(
            page_content=combined_text,
            metadata={
                "table_name": schema["table"],
                "table_type": schema["type"],
                #"full_schema": schema["columns"] # store only cols
            }
        )
    )

schema_vectorstore = create_vector_store(
    schema_docs_for_vectorstore,
    embeddings,
    persist_directory="./db_schemas"
)

print("Schema vector store created at ./db_schemas")