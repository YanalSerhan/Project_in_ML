import re
import json

def clean_query(query):
  # remove nikud
  query = re.sub(r"[\u0591-\u05C7]", "", query)
  # remove punctuation
  query = re.sub(r"[^0-9א-ת ]", " ", query)
  # collapse spaces
  query = re.sub(r"\s+", " ", query).strip()
  return query

def query_enhancement(query, query_enhancer, conv_state=None, slot_filler=None):
  # clean query
  query = clean_query(query)

  extracted = slot_filler.extract(query)
  if conv_state:
    conv_state.update(extracted)

  # rewrite the query
  rewritten = query_enhancer.rewrite(query, conv_state)

  # extract metadata
  metadata = query_enhancer.keyword_extraction(rewritten)

  # load it as json
  metadata = json.loads(metadata)

  return rewritten, metadata, conv_state