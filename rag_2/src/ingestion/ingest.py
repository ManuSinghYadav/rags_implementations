from dotenv import load_dotenv

from rag_2.src.ingestion.parsing import parsing_of_docs
from rag_2.src.ingestion.chunking import chunking_of_pdf, chunking_of_excel
from rag_2.src.ingestion.embeddings import store_emeddings
from rag_2.src.ingestion.bm25s_indexer import generating_bm25_index


def run_ingestion():
    print("Parsing and chunking documents...")
    md_text, financial_data = parsing_of_docs()
    pdf_chunks = chunking_of_pdf(md_text)
    excel_chunks = chunking_of_excel(financial_data)

    print("Generating embeddings and saving to Chroma...")
    store_emeddings(pdf_chunks, excel_chunks)

    print("Building and saving BM25S index...")
    generating_bm25_index(pdf_chunks)

    print("Ingestion complete! All database files are safely stored on disk.")


if __name__ == "__main__":
    load_dotenv(override=True)
    run_ingestion()
