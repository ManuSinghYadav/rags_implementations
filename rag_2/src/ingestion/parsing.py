import pymupdf4llm
import pandas as pd

from rag_2.src.config import APPLE_DOC_PATH, FINANCIAL_DATA_PATH


def parsing_of_docs():
    md_text = pymupdf4llm.to_markdown(str(APPLE_DOC_PATH[0]))
    financial_data = pd.read_excel(str(FINANCIAL_DATA_PATH[0]))
    return md_text, financial_data
