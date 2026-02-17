import os
from openai import OpenAI
from pathlib import Path
import json

class queryType:
    def __init__(self, model_name: str = "google/gemma-3-1b-it"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )
    def determine(self, query):
        
        BASE_DIR = Path(__file__).resolve().parent.parent
        DATA_PATH = BASE_DIR / "data" / "tables.json"
        
        structured_sources_description = ""
        # load structured sources description from data/tables.json
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            tables = json.load(f)
            for table in tables:
                structured_sources_description += f"Table: {table['type']} Describtion:\n"
                structured_sources_description += f"{table['content']}\n"
        
        prompt = f"""
You are an intent classifier for a question-answering system.

Your task is to decide whether the MAIN INTENT of a user query
can be answered using structured data sources.

Available structured data sources:
{structured_sources_description}

Classification labels:

1. "sql"
   - The main intent of the question can be answered using structured data.
   - The answer would be a table, list, or factual result from the database.
   - Extra or unsupported constraints may be safely ignored.

2. "semantic"
   - The main intent requires explanation, reasoning, opinion,
     or information not present in structured data.

DECISION RULES (VERY IMPORTANT):

- Focus ONLY on the MAIN intent of the query.
- If the main intent is answerable using structured data → "sql"
- Ignore secondary or unsupported modifiers (e.g., lecturer, time, ...)
- Do NOT require that all constraints be satisfiable.
- If the query requires explanation or subjective judgment → "semantic"
- If unsure, prefer "sql"

Output format:
Return ONLY one word: sql or semantic

User query:
"{query}"
"""


        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"thinking":False}},
            stream=True
        )
        result = ""
        for chunk in response:
          if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            result += chunk.choices[0].delta.content
        return result.strip()