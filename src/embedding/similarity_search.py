from langchain_chroma import Chroma
from embedder import embedder

def search_reviews(query_text, k=3):
    """
    Searches the ChromaDB for the most relevant reviews.
    """
    persist_path = "./chroma_db"

    # 1. Connect to the existing Database
    vector_store = Chroma(
        persist_directory=persist_path,
        embedding_function=embedder,
        collection_name="lecture_comments_collection"
    )

    # 2. Run Search
    print(f"\ Searching for: '{query_text}'")
    results = vector_store.similarity_search(query_text, k=k)

    # 3. Display Results
    print(f"--- Found {len(results)} Results ---")
    for i, doc in enumerate(results):
        lecturer = doc.metadata.get('lecturer', 'N/A')
        course = doc.metadata.get('course', 'N/A')
        content_snippet = doc.page_content[:150] # Show first 150 chars
        
        print(f"\nResult {i+1}:")
        print(f"Lecturer: {lecturer}")
        print(f"Course:   {course}")
        print(f"Review:   {content_snippet}...")

if __name__ == "__main__":
#testing
    test_query = "ביקורת על מרצים שמכניסים חומר לא רלוונטי"
    search_reviews(test_query)