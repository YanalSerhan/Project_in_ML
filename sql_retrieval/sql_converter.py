import os
from openai import OpenAI
from dotenv import load_dotenv
from sql_retrieval.table_router import cols_to_str
from pathlib import Path
import json
load_dotenv()

class SQL_converter:
  def __init__(self, model_name = "microsoft/phi-4-mini-instruct"):
    self.model_name = model_name
    self.client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_Converter_KEY")
    )


  def convert(self, query: str, metadata: dict) -> str:
    
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "tables.json"

    with open(DATA_PATH, "r", encoding="utf-8") as f:
            tables = json.load(f)

    for schema_doc in tables:
        if schema_doc["table"] == metadata['table_name']:
            cols = schema_doc["columns"]
            break
  
    #cols = get_full_schema_by_table_name(metadata['table_name'], schema_documents)
    cols = cols_to_str(cols)

    prompt = f"""
You are an expert SQL generator for an academic database.
Your task is to convert natural-language queries (Hebrew or English) into precise, safe, and accurate SQL queries.

You are given:
1. The user query
2. Table + schema
3. Constraints and semantic rules

You MUST follow ALL rules strictly.

TABLE SCHEMA:
Table name: {metadata['table_name']}
Table type: {metadata['table_type']}
Columns:
{cols}


RULES:

1. Generate exactly ONE SQL SELECT statement.
   - No comments
   - No explanations
   - No multiple queries

2. The SQL must be valid MySQL.

3. Reserved keywords MUST be escaped using backticks:
   `year`, `rank`, `avg`

4. Default SELECT behavior:
   - Use SELECT * unless the user explicitly asks for specific fields.

5. Aggregations:
   - Only apply aggregation functions (COUNT, MIN, MAX, etc.) if explicitly requested.
   - NEVER aggregate a column that is already aggregated (e.g., do NOT use AVG(avg)).

6. Ordering and limits:
   - Only use ORDER BY and LIMIT if explicitly requested by the user.
   - "highest", "top", "maximum" → ORDER BY DESC
   - "lowest", "minimum" → ORDER BY ASC

7. Filtering rules:
   - Use exact equality for named entities (e.g., course names, lecturer names).
   - Use WHERE clauses only when the user specifies filters.

8. Time semantics:
   - NEVER use CURRENT_DATE, NOW(), or system time.
   - If the user refers to "previous years" or "שנים קודמות",
     interpret this as a comparison relative to the latest available year in the data:
       `year < (SELECT MAX(year) FROM <relevant table> [WITH SAME FILTERS])`

9. Safety:
    - Do not generate UPDATE, DELETE, INSERT, DROP, or ALTER statements.

USER QUESTION:
"{query}"

Return ONLY the SQL query. Nothing else.
"""
    
    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=4096,
        stream=True
    )
    result = ""
    for chunk in response:
      if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
        result += chunk.choices[0].delta.content
    return result
