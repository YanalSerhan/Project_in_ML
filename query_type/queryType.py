import os
from openai import OpenAI


class queryType:
    def __init__(self, model_name: str = "google/gemma-3-1b-it"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("NVIDIA_API_KEY"),
            base_url="https://integrate.api.nvidia.com/v1"
        )
    def determine(self, query):
        prompt = f"""
          Classify the user query into one of two types:
            1. "sql" - if the user is asking for numeric/statistical/aggregate information 
            from the grade database (averages, distributions, list of grades, top/bottom, etc.)
            2. "semantic" - if the query is about meaning, explanations, general context,
            or conceptual questions.

            Return ONLY one word: "sql" or "semantic".

            Query: "{query}"
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"thinking":False}},
            stream=True
        )
        result = ""
        for chunk in response:
          if chunk.choices[0].delta.content is not None:
            # print(chunk.choices[0].delta.content, end="")
            result += chunk.choices[0].delta.content
        return result.strip()