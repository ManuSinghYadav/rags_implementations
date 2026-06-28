from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from rag_2.src.config import VDB_PATH

def embedding_model():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)
    return embeddings

def store_emeddings(pdf_chunks, excel_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072)

    pdf_chunk_ids = [f"pdf_chunk_{i}" for i in range(len(pdf_chunks))]
    excel_chunk_ids = [f"excel_chunk_{i}" for i in range(len(excel_chunks))]

    if VDB_PATH.exists():
        Chroma(persist_directory=str(VDB_PATH), embedding_function=embeddings).delete_collection()

    vector_store = Chroma(
        collection_name="embeddings",
        embedding_function=embeddings,
        persist_directory=str(VDB_PATH)
    )

    vector_store.add_documents(documents=pdf_chunks, ids=pdf_chunk_ids)
    vector_store.add_documents(documents=excel_chunks, ids=excel_chunk_ids)

    print(f"Vectorstore created with {vector_store._collection.count()} documents")
    