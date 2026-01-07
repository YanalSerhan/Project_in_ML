from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from util.utility import docs2str

def generate_answer(query: str, docs: list, no_info: list, sql_results: list, KB) -> str:
    # 1. Convert documents to a single context string
    context_string = ""

    if docs:
        context_string = docs2str(docs, True)
    no_info_string = ""
    if no_info:
      no_info_string = "\nthere's no info about these questions: " + docs2str(no_info)
    
    # Convert SQL results to string
    if sql_results:
        sql_string = "\nSQL Results:\n"
        for subquery, sql_answer in sql_results:
            
            sql_string += f"\nFor subquery: {subquery} ], the results are:\n"
            for row in sql_answer:
                sql_string += f"{row}\n"

            
        context_string += sql_string
    

    context = context_string + no_info_string

    #print(context)


    # 2. Initialize ChatNVIDIA model
    llm = ChatNVIDIA(model="deepseek-ai/deepseek-v3.1")

    sys_prompt = f"""
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
        - Answer the question in the same langauge as the given query.

        Your goal is to provide the most helpful answer based solely on the known data.
        """

    # 3. Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])

    # 4. Construct a LangChain Expression Language chain
    chain = prompt | llm | StrOutputParser()

    # 5. Invoke the chain and return the generated answer
    """
    answer = chain.invoke({
    "context": context,
    "question": query,
    "course": KB.courses if KB else None,
    "lecturer": KB.lecturers if KB else None,
    "year": KB.years if KB else None,
    "semester": KB.semesters if KB else None
    })
    """
    
    answer = chain.invoke({
    "context": context,
    "question": query
    })
    return answer


def generate_answer2(query, sql_result, reviews):
    """
    query: str, sql_result: list[dict], reviews: list[Document]

    return generated answer string
    """
    # 1. Convert SQL result to string
    sql_string = "SQL Results:\n"
    for row in sql_result:
        sql_string += f"course: {row['course']}, lecturer: {row['lecture']}, year: {row['year']}, semester: {row['semester']}, moed: {row['moed']}, avg: {row['avg']}\n"

    # 2. Convert reviews to context string
    reviews_string = docs2str(reviews, False)

    context = sql_string + "\nReviews:\n" + reviews_string

    print(context)

    # 3. Initialize ChatNVIDIA model
    llm = ChatNVIDIA(model="deepseek-ai/deepseek-v3.1")

    sys_prompt = f"""
        You are an answer generator for a RAG system.
        You must answer ONLY based on the provided passages.
        Do not hallucinate or invent information that does not appear in the passages.

        Given the query, the SQL results, and the reviews,
        - List the SQL results in a readable format.
        - Summarize the reviews for each lecturer and course.
        - Your answer must reference both the SQL results and the reviews.
        - Answer the question in the langauge as the given query.

        Your goal is to provide the most helpful answer based solely on the known data.
        """

    # 4. Create a ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    # 5. Construct a LangChain Expression Language chain
    chain = prompt | llm | StrOutputParser()
    # 6. Invoke the chain and return the generated answer
    answer = chain.invoke({"context": context, "question": query})
    return answer