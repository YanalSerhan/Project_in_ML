"""
Uses Hybrid Search: It executes a two-pronged search:

1. Semantic Search: Finds chunks that are conceptually similar. 
    (e.g., query "company earnings" finds a chunk about "quarterly profits").

2. Keyword Search: Finds chunks with exact matches. 
    (e.g., query "Project X-alpha" finds "Project X-alpha").

Applies Metadata Filtering: If the user asks, "What did the students say about course x and lecturer y?" 
    the retriever filters the search before doing the vector search, 
    limiting it only to chunks where course_name="x" and lecturer = "y".

Retrieves more than needed: It doesn't just retrieve 3 chunks. 
    It might retrieve 10 or 20, because the next step will filter them down.
    
This approach ensures that the most relevant chunks are considered for final selection.
"""

#from bm25_retriever import BM25Retriever
import json
from queryProcess.query_enhancement import query_enhancement, split_query, clean_json_query
from sql_retrieval.run_sql import run_sql_query
from sql_retrieval.clean_sql import clean_result
from sql_retrieval.table_router import route_query_to_table


def build_vector_filters(metadata):
    """
    metadata: dict with keys "course" and "lecturer"
    """
    filters = []
    if metadata.get("course"):
        filters.append({"course_name": {"$in": metadata['course']}})
    if metadata.get("lecturer"):
        filters.append({"lecturer": {"$in": metadata['lecturer']}})
    if not filters:
        return None
    return filters

def filter_docs(query, vector_store, filters):
    """
    filters: list of dicts with keys "course_name" and "lecturer"
    """
    if not filters:
        return docs

    relevant_docs = []
    for filter in filters:
        all = vector_store.get(where=filter)
        metadata = [data for data in all['metadatas']]
        docs = [data for data in all['documents']]

        for i in range(len(metadata)):
            relevant_docs.append((metadata[i], docs[i]))

    return docs

"""
def keyword_search(query, docs):
    # initialize BM25 instance
    retriever = BM25Retriever.from_documents(docs_split)

    # Query the retriever
    results = retriever.invoke(query)
    return results
"""

def semantic_search(subquery, vectorstore, metadata):
    """
    Detect metadata in subquery and apply AND filter for Chroma.
    """

    if not metadata:
        return vectorstore.similarity_search(
            query=subquery,
            k=10,
        )

    courses = metadata.get("course", [])
    lecturers = metadata.get("lecturer", [])

    exprs = []

    # Build individual expressions (each is a separate where-clause)
    for course in courses:
        if course in subquery:
            exprs.append({"course_name": {"$eq": course}})

    for lecturer in lecturers:
        if lecturer in subquery:
            exprs.append({"lecturer": {"$eq": lecturer}})

    # If no metadata detected → plain semantic search
    if not exprs:
        return vectorstore.similarity_search(
            query=subquery,
            k=10,
        )
    

    # If exactly one expression → don’t wrap in $and
    if len(exprs) == 1:
        where = exprs[0]
    else:
        # Proper AND filter
        where = {"$and": exprs}

    # Perform search with filter, return top 10 results
    return vectorstore.similarity_search(
            query=subquery,
            k=10,
            filter=where
        )

def classify_query(query: str, classification_vectorstore):
    results = classification_vectorstore.similarity_search_with_score(query, k=1)

    if not results:
        return {"route": "semantic", "score": None, "table": None}

    doc, _ = results[0]

    # return the same type as the closest query
    return doc.metadata["route"]

def enhanced_retrieve(query, query_enhancer, vectorstore, classification_vectorstore, sql_converter, conv_state=None, db_schemas=None):
    print("Entered enhanced_retrieve")
    
    # --- ENHANCEMENT ---
    rewritten, metadata, conv_state = query_enhancement(query, query_enhancer, conv_state)
    #splitted = query_enhancer.split_query(rewritten)
    
    splitted = split_query(rewritten)
    splitted = clean_json_query(splitted)
    splitted = json.loads(splitted)

    #extracted = slot_filler.extract(rewritten)
    #conv_state.update(extracted)
    

    print("Entered enhanced_retrieve:")
    print("rewritten:", rewritten)
    print("metadata:", metadata)
    print("splitted:", splitted)

    results_valid = []
    sql_results = []
    results_invalid = []

    for subquery in splitted:
        qtype = classify_query(subquery, classification_vectorstore)
        print(f"Subquery: {subquery} | Type: {qtype}")
        if qtype == "sql":
            print("Processing SQL subquery:", subquery)

            # decide which table to use, get the metadata
            table_metadata = route_query_to_table(subquery, db_schemas)
            print("Routed to table metadata:", table_metadata)

            # send the metadata to sql_converter along with the subquery

            sql_query = sql_converter.convert(subquery, table_metadata)
            print("Generated SQL:", sql_query)
            cleaned_sql = clean_result(sql_query)
            print("Cleaned SQL:", cleaned_sql)
            answer = run_sql_query(cleaned_sql)
            sql_results.append((subquery, answer))
            continue

        result = semantic_search(subquery, vectorstore, metadata)
        if result:
          results_valid.append(result)
        else:
          results_invalid.append(subquery)


    return results_valid, results_invalid, sql_results, conv_state