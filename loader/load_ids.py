import json

def load_ids(file_path="data/Ids.json"):
    with open(file_path, encoding="utf-8") as f:
        ids = json.load(f)
    table = next(item for item in ids if item.get("type") == "table")
    ids = table["data"]

    courses = [item["course"] for item in ids]
    lecturers = [item["lecture"] for item in ids]

    lecturers = list(set(lecturers))
    courses = list(set(courses))

    return courses, lecturers
