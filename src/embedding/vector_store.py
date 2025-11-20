import json
import os
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embedder import embedder

# FIX 1: Update the path to match your folder structure
def build_vector_store(json_file_path="data/processed/cleaned_reviews.json"):
    print(f"--- Loading data from {json_file_path} ---")

    if not os.path.exists(json_file_path):
        print(f" Error: The file '{json_file_path}' was not found.")
        print("Please check if you are running this from the root 'Project_in_ML' folder.")
        return

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    
    # FIX 2: Handle the nested structure (Course -> Reviews -> Content)
    for course_record in data:
        # Get course-level info
        c_name = course_record.get("course_name", "Unknown")
        lecturer = course_record.get("lecturer", "Unknown")
        
        # Loop through the list of reviews for this specific course
        reviews_list = course_record.get("reviews", [])
        
        for review in reviews_list:
            content = review.get("content")
            
            # Create metadata for this specific review
            metadata = {
                "course": c_name,
                "lecturer": lecturer,
                "rank": review.get("rank"),
                "time": review.get("time")
            }

            if content:
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

    print(f" Prepared {len(documents)} documents for embedding.")

    if len(documents) == 0:
        print(" Warning: No documents found. Check your JSON keys.")
        return

    # 3. Reset and Create ChromaDB
    persist_path = "./chroma_db"
    
    if os.path.exists(persist_path):
        shutil.rmtree(persist_path)

    print("Embedding documents... (This may take a moment)")
    
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedder,
        collection_name="lecture_comments_collection",
        persist_directory=persist_path
    )
    
    print(f" Success! Vector store saved to '{persist_path}'")

if __name__ == "__main__":
    build_vector_store()