from retrieval.retrieval import enhanced_retrieve
from generation.answerGenerator import generate_answer
#from reranker.reranker import rerank_documents

def RAG(query, query_enhancer, vectorstore, sql_converter, conv_state=None, db_schema=None):
  print("Entered RAG File")
  results_valid, results_invalid, sql_results, KB = enhanced_retrieve(query, query_enhancer, 
                                                                  vectorstore, 
                                                                  sql_converter, conv_state=conv_state,
                                                                  db_schemas=db_schema)
  #reranked_docs = []
  if results_valid:
    #reranked_docs = rerank_documents(query, results_valid[0]) // returns max 5 documents
    print("Length of valid docs:", len(results_valid))
    print()
  #answer = generate_answer(query, reranked_docs, results_invalid, sql_results, KB)
  answer = generate_answer(query, results_valid, results_invalid, sql_results)
  return answer