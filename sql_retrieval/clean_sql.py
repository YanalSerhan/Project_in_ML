def clean_result(result):
  # remove ```sql and \n
  result = result.replace("```", "")
  result = result.replace("sql", "")
  result = result.replace("\n", " ")
  return result