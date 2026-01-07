from retrieval.retrieval import enhanced_retrieve
from generation.generation import generate_answer
from reranker.reranker import rerank_documents

def RAG(query, query_enhancer, vectorstore, courses, lecturers, queryTyper, sql_converter, conv_state=None, slot_filler=None, db_schema=None):
  results_valid, results_invalid, sql_results, KB = enhanced_retrieve(query, query_enhancer, 
                                                                  vectorstore, queryTyper, 
                                                                  sql_converter, conv_state=conv_state, slot_filler=slot_filler,
                                                                  db_schemas=db_schema)
  #print(results_valid)
  #print(results_invalid)
  reranked_docs = []
  if results_valid:
    reranked_docs = rerank_documents(query, results_valid[0])
  answer = generate_answer(query, reranked_docs, results_invalid, sql_results, KB)
  return answer