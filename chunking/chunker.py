from langchain_text_splitters import RecursiveCharacterTextSplitter # Updated import

def chunk_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap= 50,
        separators=["\n\n", "\n", ".", ";", ",", " ", ""],
    )

    ## Some nice custom preprocessing
    # documents[0].page_content = documents[0].page_content.replace(". .", "")
    docs_split = text_splitter.split_documents(docs)

    # assign chunk id
    for i, doc in enumerate(docs_split):
        doc.metadata["id"] = i

    return docs_split