import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    
    """
    Function that takes list of the format:
    
    {
        "course_id": int,
        "course_name": str,
        "lecturer": str,
        "reviews": List[dict{"content": str, "rank": int, "time": str}],
        "metadata": dict{"tag": str, "num_reviews": int}
    }

    and returns list of chunked texts with metadata.
    """
    def chunk_courses(self, entry: List[Dict]) -> List[Dict]:
        chunked_texts = []
        
        for course in entry:
            course_id = course.get("course_id")
            course_name = course.get("course_name")
            lecturer = course.get("lecturer")
            reviews = course.get("reviews", [])
            metadata = course.get("metadata", {})
            
            for review in reviews:
                content = review.get("content", "")
                rank = review.get("rank")
                time = review.get("time")
                
                chunks = self.text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    chunked_texts.append({
                        "text": chunk,
                        "metadata": {
                            "course_id": course_id,
                            "course_name": course_name,
                            "lecturer": lecturer,
                            "rank": rank,
                            "time": time,
                            "chunk_index": i,
                            **metadata
                        }
                    })
        
        return chunked_texts
    