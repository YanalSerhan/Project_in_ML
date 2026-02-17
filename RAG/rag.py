from retrieval.retrieval import enhanced_retrieve
from generation.generation import generate_answer
from reranker.reranker import rerank_documents

def RAG(query, query_enhancer, vectorstore, classification_db, sql_converter, conv_state=None, db_schema=None):
  print("Entered RAG File")
  results_valid, results_invalid, sql_results, KB = enhanced_retrieve(query, query_enhancer, 
                                                                  vectorstore, classification_db, 
                                                                  sql_converter, conv_state=conv_state,
                                                                  db_schemas=db_schema)
  #print(results_valid)
  #print(results_invalid)
  reranked_docs = []
  if results_valid:
    reranked_docs = rerank_documents(query, results_valid[0])
  answer = generate_answer(query, reranked_docs, results_invalid, sql_results, KB)
  return answer