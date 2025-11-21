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