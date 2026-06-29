import bm25s
import numpy as np
from langchain_core.documents import Document

from rag_2.src.config import BM25S_PATH

# 2. Natively load your BM25S index from your storage folder
bm25s_retriever = bm25s.BM25.load(BM25S_PATH, load_corpus=True)


def bm25_retriever_search(question: str, file_type_filter=None, k=20):
    question_tokens = bm25s.tokenize(texts=question)
    
    mask = None
    if file_type_filter:
        mask = np.array([doc["metadata"].get("file_type") == file_type_filter for doc in bm25s_retriever.corpus]).astype(bool)
        
    results, _ = bm25s_retriever.retrieve(question_tokens, k=k, weight_mask=mask)
    
    # Map the stored dictionaries back into LangChain Document instances for your RRF function
    langchain_docs = [
        Document(page_content=res_dict["text"], metadata=res_dict["metadata"])
        for res_dict in results[0]
    ]
    
    return langchain_docs
