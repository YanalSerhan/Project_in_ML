from preprocessing.text_cleaner import Cleaner

class QueryProcessor:
    def __init__(self,llm):
        self.llm = llm

    def clean_query(self, query):
        return Cleaner("").clean_hebrew_text(query)
    
    def keywords_extraction(self, query):
        prompt = f"Extract keywords from the following query:\n\n{query}\n\nKeywords:"
        response = self.llm.generate(prompt)
        keywords = response.text.strip().split(", ")
        return keywords
    
    def llm_rewrite(self, query):
        prompt = f"""rewrite the following query into a more effective search query
        that will help retrieve relevant course reviews from a database. 
        The rewritten query should be concise and focus on the main topics/aspects mentioned in the original query.
        Expand abbreviations and slang where necessary.
        Original Query: {query}
        """
    
        response = self.llm.generate(prompt)
        rewritten_query = response.text.strip()
        return rewritten_query
    
    def process_query(self, query):
        cleaned_query = self.clean_query(query)
        rewritten_query = self.llm_rewrite(cleaned_query)
        keywords = self.keywords_extraction(rewritten_query)
        return {
            "rewritten_query": rewritten_query,
            "keywords": keywords
        }