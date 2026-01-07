def route_query_to_table(query: str, schema_vectorstore):
    """
    Routes a user query to the most relevant table schema using a vector store.
    """
    # Perform a similarity search to find the most relevant schema document
    # We'll retrieve the top 1 result as we expect one primary table for a query.
    docs = schema_vectorstore.similarity_search(query, k=1)

    if not docs:
        return None

    # The most relevant document should be the first one
    relevant_doc = docs[0]
    return relevant_doc.metadata

# convert dict to str
def cols_to_str(cols):
    res = ""
    for col in cols:
        res += f"- {col["col"]}: {col["description"]}\n"
    return res

def get_full_schema_by_table_name(table_name: str, schema_docs: list) -> dict:
    """
    Retrieves the cols for a given table name
    from the list of schema documents.
    """
    for schema_doc in schema_docs:
        if schema_doc["table"] == table_name:
            return schema_doc["columns"]
    return []