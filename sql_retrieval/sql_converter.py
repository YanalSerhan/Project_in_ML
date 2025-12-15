import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class SQL_converter:
  def __init__(self, model_name = "microsoft/phi-4-mini-instruct"):
    self.model_name = model_name
    self.client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_Converter_KEY")
    )


  def convert(self, query):
    prompt = f"""
      You are an expert SQL generator for an academic system.
      Your task is to convert natural-language queries (Hebrew or English) into precise, safe, and accurate SQL queries.

      Important rules:
      1. Use ONLY the following table:
        Table: q
          - id (INT)
          - course (VARCHAR)
          - lecture (VARCHAR)
          - year (INT)
          - semester (VARCHAR)
          - moed (VARCHAR)
          - avg (FLOAT)
          - tag (VARCHAR)
          - proj (VARCHAR)
          - rank (INT)

      2. NEVER use AVG() on avg — it is already an average.
        When the user asks for “highest average”, return the single row where avg is the highest.

      3. If the user ask for high number (e.g. > 10) of results, use LIMIT 10 to restrict the number of returned rows.

      4. If the user mentions a course name (e.g., "מבני נתונים"), search it using:
          WHERE course = 'מבני נתונים'.
         If the user mentions a lecturer name (e.g., "אורן וימן"), search it using:
          WHERE lecture = 'אורן וימן'.

      5. If the user asks about:
        - a year → include `WHERE year = X`
        - moed → include `WHERE moed = 'a'`, 'b', etc.
        - "highest" / "biggest" / "top" → use ORDER BY DESC + LIMIT 1
        - "lowest" → use ORDER BY ASC + LIMIT 1

      6. Output **ONLY SQL code**, no explanations, no comments.

      7. The SQL must be fully executable in MySQL.

      8. Always start the query with: SELECT * FROM q unless specified otherwise.
      if specified otherwise, use Escaping with backticks on reserved keywords like `year`, `avg`, `rank`.

      Your job: Convert the user query into a **single, correct SQL SELECT statement** following these rules.


      User question:
      "{query}"

      Return only SQL code nothing else.
    """
    response = self.client.chat.completions.create(
        model=self.model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        top_p=0.7,
        max_tokens=2048,
        stream=True
    )
    result = ""
    for chunk in response:
      if chunk.choices[0].delta.content is not None:
        # print(chunk.choices[0].delta.content, end="")
        result += chunk.choices[0].delta.content
    return result
