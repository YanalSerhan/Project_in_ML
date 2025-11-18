from typing import List, Dict, Any
from preprocessing.group_by_id import group_by_id
from preprocessing.text_cleaner import Cleaner
import json

path = "../data/preprocessed/preproComm.json"

# Group by id
grouped_data = group_by_id(path).group()

# Cleaning
cleaner = Cleaner("")
for course in grouped_data:
    for review in course["reviews"]:
        review["content"] = cleaner.clean_hebrew_text(review["content"])

# Save cleaned data
with open("cleaned_reviews.json", "w", encoding="utf-8") as f:
    json.dump(grouped_data, f, ensure_ascii=False, indent=2)

print("Cleaned data saved to cleaned_reviews.json")