"""
The initial retrieval is fast but not always perfectly accurate. This step re-scores the retrieved chunks to find the best ones.
Uses a Re-Ranker: It takes the Top-K (e.g., 20) retrieved chunks and passes them to a more powerful, slower model
(like a Cross-Encoder) that directly compares the query to each chunk and assigns a new, more accurate relevance score.

Selects the Final Context: It picks the new Top-N (e.g., 3-5) chunks based on the re-ranker's scores.

Manages Context: It intelligently formats these chunks into a final block of text to send to the LLM, 
often placing the most relevant chunks at the very beginning and end of the context 
(to avoid the "lost in the middle" problem where LLMs ignore information in the middle of a large prompt).

This re-ranking step significantly improves the quality of the context provided to the LLM, leading to better answers.
"""