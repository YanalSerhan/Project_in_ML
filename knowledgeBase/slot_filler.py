import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


EXTRACTION_PROMPT = """
You are an information extraction system.
Task:
Extract the *course name(s)*, *lecturer name(s)*, years and semesters mentioned in the user query.

Return JSON with the following format:

{
  "courses": [string],
  "lecturers": [string],
  "years": [int],
  "semesters": [string]
}

Rules:
- If nothing is mentioned for a field, return an empty list for that field.
- Do NOT guess. Identify only explicit mentions.
- No hallucinations.
- Strict JSON only.
"""

class SlotFiller:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-v3.1"):
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )
        self.model_name = model_name

    def extract(self, query: str) -> dict:
        print("Entered Slot Filler File")
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": EXTRACTION_PROMPT + "\nUser query:\n" + query}],
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
        response = result.strip()
        try:
            data = json.loads(response)
        except:
            # safe output
            return {"courses": [], "lecturers": [], "years": [], "semesters": []}

        # ensure all keys exist
        for key in ["courses", "lecturers", "years", "semesters"]:
            data.setdefault(key, [])

        print("finished Slot Filler File")
        return data