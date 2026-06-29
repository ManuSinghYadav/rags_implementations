from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag_2.src.config import CONFIG, VDB_PATH


load_dotenv()

# Natively load the persistent vector store from your storage folder
embeddings = OpenAIEmbeddings(
    model=CONFIG["embedding_model"], dimensions=CONFIG["embedding_size"]
)
vector_store = Chroma(
    persist_directory=str(VDB_PATH),
    embedding_function=embeddings,
    collection_name="embeddings",  # Ensure consistent collection name as during ingestion
)


def retrieve_filtered_vector_docs(query: str, intent):

    search_filter = None
    if intent.has_filter and intent.file_type:
        search_filter = {"doc_type": intent.file_type}

    docs = vector_store.similarity_search(
        query=query, k=CONFIG["vdb_top_k"], filter=search_filter
    )

    return docs
