import os
import re
import json
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# 1. Get the absolute path to the directory containing THIS script (query_enhancement)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the path: go up one level ('..'), then into the 'nicknames' folder
NICKNAMES_FILE_PATH = os.path.join(CURRENT_DIR, '..', 'nicknames', 'course_nicknames.json')

# 3. Safely load the file, with a fallback just in case the path is slightly off
try:
    with open(NICKNAMES_FILE_PATH, 'r', encoding='utf-8') as f:
        NICKNAMES_MAP = json.load(f)
except FileNotFoundError:
    print(f"⚠️ Warning: Could not find the file at {NICKNAMES_FILE_PATH}")
    NICKNAMES_MAP = {} # Fallback to empty dict so the code doesn't completely crash

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

def update_conv_state(conv_state, metadata):
    for key, value in metadata.items():
        if key in conv_state:
            conv_state[key].append(value)
        else:
            conv_state[key] = [value]

def query_enhancement(query, query_enhancer, conv_state=None):
  print("Entered query_enhancement")
  # clean query
  query = clean_query(query)

  conv_state_str = to_prompt_str(conv_state) if conv_state else ""
  result = query_enhancer.rewrite_and_extract(query, conv_state_str)
  result = json.loads(result)
  rewritten = result["rewritten_query"]
  metadata = {"course": result["course"], "lecturer": result["lecturer"]}

  # Update rewritten query with official course names
  for i, old_name in enumerate(metadata["course"]):
      new_name = NICKNAMES_MAP.get(old_name, old_name)
      if old_name in rewritten:
        rewritten = rewritten.replace(old_name, new_name)
      metadata["course"][i] = new_name

  if conv_state:
    print("Updating conversation state with new metadata:", conv_state)
    update_conv_state(conv_state, metadata)
  
  return rewritten, metadata, conv_state

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
def split_query(query: str) -> str:
  prompt = f"""
You are a query decomposition assistant.

Your task is to transform a user query into ONE OR MORE simpler sub-queries
that can be answered independently and later combined.

Core Principle:
Split ONLY when the user explicitly compares or mentions TWO OR MORE distinct named entities
(e.g., two lecturers, two companies, multiple courses).

Definition of "distinct entity":
A proper noun or clearly separable named alternative (e.g., "Microsoft or Google",
"יורי או נגה", "Course A and Course B").

DO NOT split when:
- The query refers to only ONE entity.
- The query contains multiple aspects of the SAME entity.
- The query is complex but not comparing multiple alternatives.

Decomposition Rules:

1. If multiple entities are compared → create ONE sub-query per entity.
2. Each sub-query must:
   - Preserve the full original context (course name, year, metric, etc.).
   - Focus on ONLY ONE entity.
   - Be a standalone, grammatically correct question.
   - Keep the original intent (do NOT change "profit" to "revenue" unless explicitly stated).
3. Keep the original language of the query.
4. Do NOT answer the question.
5. Do NOT add explanations.
6. Output ONLY a valid JSON list of strings.
7. If no split is required → return a JSON list containing the original query unchanged.
8. SPLIT ON MULTIPLE ENTITIES: If the query compares or mentions TWO or more lecturers -> produce one sub-query per lecturer.

Good Examples:

Input:
Did Microsoft or Google make more profit last year?

Output:
[
  "How much profit did Microsoft make last year?",
  "How much profit did Google make last year?"
]

Input:
עם מי עדיף לעשות את הקורס תכנון וניתוח אלגוריתמים, יורי או נגה?

Output:
[
  "האם עדיף לעשות את הקורס תכנון וניתוח אלגוריתמים עם יורי?",
  "האם עדיף לעשות את הקורס תכנון וניתוח אלגוריתמים עם נגה?"
]

Now decompose the following query:

{query}
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