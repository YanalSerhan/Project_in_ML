import os
from openai import OpenAI, APITimeoutError
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from util.utility import docs2str

def generate_answer(
    query: str,
    docs: list,
    no_info: list,
    sql_results: list,
    KB = None,
    model_name: str = "gpt-4o-mini",  # change to any OpenAI model you prefer
) -> str:
    """
    Generates answers for RAG using OpenAI API.
    Keeps the exact same logic as the original class version.
    """

    print("Entered Generation File")

    # ------------------------
    # 1️⃣ Build Context String
    # ------------------------
    context_string = ""

    if docs:
        context_string = docs2str(docs, True)

    no_info_string = ""
    if no_info:
        no_info_string = "\nthere's no info about these questions: " + docs2str(no_info)

    if sql_results:
        sql_string = "\nSQL Results:\n"
        for subquery, sql_answer in sql_results:
            sql_string += f"\nFor subquery: {subquery}], the results are:\n"
            for row in sql_answer:
                sql_string += f"{row}\n"

        context_string += sql_string

    context = context_string + no_info_string

    # ------------------------
    # 2️⃣ System Prompt
    # ------------------------
    sys_prompt = """
You are an answer generator for a RAG system.
You must answer ONLY based on the provided passages.
Do not hallucinate or invent information that does not appear in the passages.

- Never hallucinate unavailable information.
If the user asks to compare or evaluate multiple lecturers/courses:
- Summarize what information DOES appear in the passages.
- Explicitly state when information about a lecturer or course is NOT present,
  and state that the reason for that is maybe because the lecturer doesn't teach the course
  or something like that.
- Never answer “אין תשובה” or “לא מצאתי מידע” if partial information exists.
- Given also the averages of the last 6 exams for each chunk in the format: avg, year, semester, moed.
- Answer the question in the same language as the given query.

Your goal is to provide the most helpful answer based solely on the known data.
"""

    # ------------------------
    # 3️⃣ Full Prompt
    # ------------------------
    full_prompt = f"""
{sys_prompt}

Context:
{context}

Question:
{query}

Answer:
"""

    # ------------------------
    # 4️⃣ OpenAI Client
    # ------------------------
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=20.0,
    )

    # ------------------------
    # 5️⃣ Retry + Streaming Call
    # ------------------------
    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
    )
    def _call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=8192,
            stream=True,
        )

        result = ""
        for chunk in response:
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content is not None
            ):
                result += chunk.choices[0].delta.content

        return result.strip()

    answer = _call_llm(full_prompt)

    print("Finished generation")
    return answer