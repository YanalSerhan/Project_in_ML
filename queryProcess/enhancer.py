import os
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
load_dotenv()

class QueryEnhancer:
    """
    Rewrites and expands user queries to improve retrieval quality in RAG.
    Uses an NVIDIA NIM LLM (e.g., mistral, llama3, etc.)
    """

    def __init__(self, model_name: str = "mistralai/mixtral-8x7b-instruct-v0.1"):
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.model_name = model_name

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
    def _call_llm(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking":False}},
            stream=True
        )
        result = ""
        for chunk in response:
          if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
            result += chunk.choices[0].delta.content
        return result.strip()

    def rewrite(self, query: str, KB) -> str:
        """
        Sends the query to an NIM model to get a rewritten/expanded version.
        """
        print("Entered rewrite")
        state_info = ""
        if KB:
            state_info = KB.to_prompt_str()
        prompt = f"""
          Rewrite the following user query to be clearer, more explicit, and easier for a retrieval system to understand.
          Keep the meaning exactly the same. Do not add new information or assumptions.
          If the query is ambiguous, rewrite it in a neutral and generic way without resolving the ambiguity.
          Do NOT guess missing details.
          Never infer ownership, perspective, or subject unless it explicitly appears in the user query.
          Fix spelling mistakes and expand abbreviations only when the meaning is obvious.
          Some lecturer names comes with their surrnames and some don't; keep them as it is.
          Keep the rewritten query short and focused.
          If the query has references like "the lecturer," "this course," etc., which are ambiguous without context, try
          to infer them from the knowledge base: {state_info}. if you cannot infer them, keep them as is.
          Example for abbreviations: algo -> algorithms, DS -> data structure.

          Example for rewriting:
          User query: "What is the avg grade in algro b and who is the lecturer?"
          Rewritten query: "What is the average grade in the course Algebra B and who is the lecturer for that course?"

          Do NOT answer the query.
          Do NOT provide explanations.
          Return ONLY the rewritten query in the same language as the input.

          User query:
          {query}

          Rewritten query:
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking":False}},
            stream=True
        )
        result = ""
        for chunk in response:
          if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            result += chunk.choices[0].delta.content
        
        print("finished rewrite")
        return result.strip()

    def keyword_extraction(self, query: str) -> str:
        """
        Sends the query to an NIM model to get a rewritten/expanded version.
        """
        print("Entered keyword_extraction")
        prompt = f"""
        You are an information extraction system.
        Task:
        Extract the *course name(s)* and *lecturer name(s)* mentioned in the user query below.
        Query:
        "{query}"

        Output format (strict JSON only):
        {{
          "course": list of course names,
          "lecturer": list of lecturer names
        }}

        Rules:
        - If no course is mentioned, return "course": [].
        - If no lecturer is mentioned, return "lecturer": [].
        - Do NOT infer or guess; only extract if explicitly stated.
        - Return ONLY the JSON. No explanations.
        - Some courses names might include versions, for example Algebra b, extract them as well.
        - Some lecturer names comes with their surrnames and some don't; extract what appears in the query.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking":False}},
            stream=True
        )
        result = ""
        for chunk in response:
          if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            result += chunk.choices[0].delta.content
        print("finished keyword_extraction")
        return result.strip()

    def split_query(self, query: str) -> str:
        """
        If a user asks, "What's the difference in revenue and user growth between Q1 and Q2?"
        it splits this into multiple queries:
        "Q1 revenue," "Q2 revenue," "Q1 user growth," and "Q2 user growth."
        """
        print("Entered split_query")
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

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            extra_body={"chat_template_kwargs": {"thinking":False}},
            stream=True
        )
        result = ""
        for chunk in response:
          if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            result += chunk.choices[0].delta.content
        print("finished split_query")
        return result
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
    def rewrite_and_extract(self, query: str):
      state_info = ""
      prompt = f"""
You are a query rewriting and information extraction system.

Your tasks:

1) Rewrite the user query to be clearer, more explicit, and easier for a retrieval system to understand.
2) Extract the course name(s) and lecturer name(s) explicitly mentioned in the query.

REWRITING RULES:
- Keep the meaning exactly the same.
- Do NOT add new information or assumptions.
- If the query is ambiguous, rewrite it in a neutral and generic way without resolving the ambiguity.
- Do NOT guess missing details.
- Never infer ownership, perspective, or subject unless it explicitly appears in the user query.
- Fix spelling mistakes and expand abbreviations only when the meaning is obvious.
- Example abbreviations: algo -> algorithms, DS -> data structure.
- Some lecturer names come with surnames and some don't; keep them exactly as written.
- Keep the rewritten query short and focused.
- If the query has references like "the lecturer," "this course," etc., which are ambiguous without context, try to infer them from the knowledge base: {state_info}. If you cannot infer them, keep them as is.
- Do NOT answer the query.
- Do NOT provide explanations.
- Return the output in the same language as the input.

EXTRACTION RULES:
- Extract ONLY course name(s) and lecturer name(s) explicitly mentioned in the query.
- Do NOT infer or guess anything.
- If no course is mentioned, return an empty list.
- If no lecturer is mentioned, return an empty list.
- Some course names may include versions (e.g., Algebra B); extract them fully.
- Some lecturer names may include surnames or only first names; extract exactly what appears.

Output format (STRICT JSON only, no explanations):

{{
  "rewritten_query": "...",
  "course": [],
  "lecturer": []
}}

User query:
"{query}"
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