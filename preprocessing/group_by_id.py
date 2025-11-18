import json
from collections import defaultdict

class group_by_id:
    """
    input: json file with list of dicts of the format:
    {
        "course_id": str,
        "course_name": str,
        "lecturer": str,
        "reviews": List[dict{"content": str, "rank": int, "time": str}],
        "metadata": dict{"tag": str, "num_reviews": int}
    }
    output: dict grouped by course_id
    """
    def __init__(self, path_to_json):
        self.path_to_json = path_to_json
        
    
    def group(self):
        with open(self.path_to_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract table entry
        table = next(item for item in data if item.get("type") == "table")
        rows = table["data"]

        # Group by id
        grouped = defaultdict(lambda: {
            "course_name": None,
            "lecturer": None,
            "reviews": [],
            "metadata": {}
        })

        for entry in rows:
            cid = entry["id"]

            grouped[cid]["course_name"] = entry["course"]
            grouped[cid]["lecturer"] = entry["lecture"]
            currReview = {"content":entry["content"], "rank":entry["rank"], "time": entry["time"]}
            grouped[cid]["reviews"].append(currReview)

            # add metadata
            grouped[cid]["metadata"]["tag"] = entry["tag"]
            #grouped[cid]["metadata"]["rank"] = entry["rank"]

        # Convert to final list
        final_dataset = []
        for cid, info in grouped.items():
            final_dataset.append({
                "course_id": cid,
                "course_name": info["course_name"],
                "lecturer": info["lecturer"],
                "reviews": info["reviews"],
                "metadata": {
                    "tag": info["metadata"]["tag"],
                    #"rank": info["metadata"]["rank"],
                    "num_reviews": len(info["reviews"])
                }
            })
        return final_dataset
                

            