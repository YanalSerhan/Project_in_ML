import json
from langchain_core.documents import Document


def load_reviews(reviews_path = "data\\cleaned_reviews.json"):
    with open("data\\cleaned_reviews.json", encoding="utf-8") as f:
        data = json.load(f)

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
    return docs