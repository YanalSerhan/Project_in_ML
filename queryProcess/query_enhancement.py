import re
import json
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

def clean_query(query):
  # remove nikud
  query = re.sub(r"[\u0591-\u05C7]", "", query)
  # remove punctuation
  query = re.sub(r"[^0-9א-ת ]", " ", query)
  # collapse spaces
  query = re.sub(r"\s+", " ", query).strip()
  return query

def to_prompt_str(knowledge_base: dict):
    prompt_str = ""
    for key, values in knowledge_base.items():
        prompt_str += f"{key.capitalize()} mentioned in conversation: {values}\n"
        prompt_str += f"Most recent {key} (used for vague references): {values[-1] if values else 'None'}\n"
    return prompt_str

def query_enhancement(query, query_enhancer, conv_state=None):
  print("Entered query_enhancement")
  # clean query
  query = clean_query(query)

  conv_state_str = to_prompt_str(conv_state) if conv_state else ""
  result = query_enhancer.rewrite_and_extract(query, conv_state_str)
  result = json.loads(result)
  rewritten = result["rewritten_query"]
  metadata = {"course": result["course"], "lecturer": result["lecturer"]}

  if conv_state:
    conv_state.update(metadata)
  
  return rewritten, metadata, conv_state

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
def split_query(query: str) -> str:
  prompt = f"""
        You are a helpful assistant that prepares queries that will be sent to a search component.
        Sometimes, these queries are very complex.
        Your job is to simplify complex queries into multiple queries that can be answered in isolation to eachother.

        - Only split the query when the user explicitly mentions MULTIPLE distinct entities (e.g., two lecturers, two courses, multiple people, multiple items).
        - If the query mentions only ONE entity and ONE topic, do NOT split — return the original query as a single element.
        - Use the same language as the query.
        - Make the questions make sense.

        Rules:
        1. If the query contains multiple lecturers → produce one sub-query per lecturer.
        2. If the query contains multiple courses → produce one sub-query per course.
        3. If there is only ONE lecturer and ONE course → return the query unchanged.
        4. Output format must be ONLY a JSON list of strings.


        Examples
        1. Query: Did Microsoft or Google make more money last year?
          ['How much profit did Microsoft make last year?', 'How much profit did Google make last year?']
        2. Query: What is the capital of France?
          ['What is the capital of France?']
        3. Query: {query}
          ['<subquery>', ...]
        """

  client = OpenAI()
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
       {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

  res = response.choices[0].message.content
  return res

def clean_json_query(query):
  """
  format: 
    ```json
        ["<subquery>", ...]
    ```
  """
  query = query.replace("```", "")
  query = query.replace("json", "")
  query = query.replace("\n", "")
  return query