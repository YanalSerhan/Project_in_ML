import json
import os
from typing import List, Dict, Any, Optional


class JSONLoader:
    """
    Loads JSON documents for the RAG ingestion pipeline.

    Expected JSON structure example:
    {
        {
        "course_id": int,
        "course_name": str,
        "lecturer": str,
        "reviews": List[dict{"content": str, "rank": int, "time": str}],
        "metadata": dict{"tag": str, "num_reviews": int}
        }
    }
    """

    def __init__(self, raw_data_dir: str):
        """
        :param raw_data_dir: Path to directory containing *.json files.
        """
        self.raw_data_dir = raw_data_dir

    def load_file(self, path: str) -> Optional[Dict[str, Any]]:
        """
        Load a single JSON file.

        :param path: File path to load.
        :return: Parsed JSON dictionary or None if failed.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure required fields exist
            if "text" not in data:
                raise ValueError(f"JSON file {path} must contain a 'text' field.")

            # Auto-add id if missing
            if "id" not in data:
                data["id"] = os.path.basename(path).replace(".json", "")

            # Ensure metadata exists
            data.setdefault("metadata", {})
            data["metadata"].setdefault("source_file", os.path.basename(path))

            return data

        except Exception as e:
            print(f"[ERROR] Failed to load JSON file {path}: {e}")
            return None

    def load_all(self) -> List[Dict[str, Any]]:
        """
        Load all JSON files in the raw data directory.

        :return: List of JSON documents.
        """
        documents = []

        for filename in os.listdir(self.raw_data_dir):
            if filename.lower().endswith(".json"):
                file_path = os.path.join(self.raw_data_dir, filename)
                doc = self.load_file(file_path)

                if doc:
                    documents.append(doc)

        return documents


# Convenience function
def load_json_documents(raw_data_dir: str) -> List[Dict[str, Any]]:
    """
    Short helper for pipelines.

    Example:
        docs = load_json_documents("data/raw/")
    """
    loader = JSONLoader(raw_data_dir)
    return loader.load_all()

"""
# Example usage
if __name__ == "__main__":
    raw_data_directory = "data/raw/"
    documents = load_json_documents(raw_data_directory)
    for doc in documents:
        print(f"Loaded document ID: {doc['id']} with title: {doc.get('title', 'N/A')}")

"""