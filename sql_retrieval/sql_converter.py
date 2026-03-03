import os
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from sql_retrieval.table_router import cols_to_str
from pathlib import Path
import json

load_dotenv()

class SQL_converter:
    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(7),
        retry=retry_if_exception_type(Exception)
    )
    def convert(self, query: str, metadata: dict) -> str:

        BASE_DIR = Path(__file__).resolve().parent.parent
        DATA_PATH = BASE_DIR / "data" / "tables.json"

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            tables = json.load(f)

        cols = None
        for schema_doc in tables:
            if schema_doc["table"] == metadata['table_name']:
                cols = schema_doc["columns"]
                break

        if cols is None:
            raise ValueError(f"Table {metadata['table_name']} not found in schema")

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
   `year`, `avg`

4. Default SELECT behavior:
   - Use SELECT * always.

5. Aggregations:
   - Only apply aggregation functions (COUNT, MIN, MAX, etc.) if explicitly requested.
   - NEVER aggregate a column that is already aggregated (e.g., do NOT use AVG(avg)).

6. Ordering and limits:
   - Only use ORDER BY and LIMIT if explicitly requested.
   - "highest", "top", "maximum" → ORDER BY DESC
   - "lowest", "minimum" → ORDER BY ASC

7. Filtering rules:
   - Use exact equality for named entities.
   - Use WHERE clauses only when the user specifies filters.

8. Time semantics:
   - NEVER use CURRENT_DATE, NOW(), or system time.
   - If the user refers to "previous years" or "שנים קודמות",
     interpret this as:
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
            max_tokens=8192
        )

        result = response.choices[0].message.content.strip()

        return result