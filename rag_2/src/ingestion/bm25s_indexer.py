import bm25s

from rag_2.src.config import BM25S_PATH


def generating_bm25_index(chunks):
    dict_corpus = [
        {
            "text": doc.page_content,
            "metadata": doc.metadata  # Keeps your chunk_id and file_type intact
        }
        for doc in chunks
    ]

    corpus_text = [doc["text"] for doc in dict_corpus]
    corpus_tokens = bm25s.tokenize(texts=corpus_text)
    bm25_retriever = bm25s.BM25(corpus=dict_corpus)
    bm25_retriever.index(corpus_tokens)
    bm25_retriever.save(BM25S_PATH)
