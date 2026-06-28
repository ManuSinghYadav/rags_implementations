from langchain_text_splitters import MarkdownHeaderTextSplitter	
from langchain_core.documents import Document


def chunking_of_pdf(md_text):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
    pdf_chunks = text_splitter.split_text(md_text)

    # Since the splitter alreadt wraps the chunks in Document class, we can add more metadata to each chunk

    pdf_chunk_ids = [f"pdf_chunk_{i}" for i in range(len(pdf_chunks))]

    for index, chunks in enumerate(pdf_chunks):
        chunks.metadata["source"] = "apple_10k.md"
        chunks.metadata["doc_type"] = "PDF-Markdown"
        chunks.metadata["chunk_id"] = pdf_chunk_ids[index]
    
    return pdf_chunks


def chunking_of_excel(financial_data):
    # Row-as-a-string method
    excel_chunks = []
    excel_chunk_ids = [f"excel_chunk_{i}" for i in range(len(financial_data))]

    for index, row in financial_data.iterrows():
        row_text = (
                f"In {row['Month Name']} {row['Year']}, the {row['Segment']} segment in {row['Country']} "
                f"purchased {row['Units Sold']:,} units of the product '{row['Product']}'. "
                f"The manufacturing price was ${row['Manufacturing Price']} and the sale price was ${row['Sale Price']}. "
                f"This transaction generated ${row['Gross Sales']:,} in gross sales with a discount band of ${row['Discount Band']} "
                f"amounting to ${row['Discounts']:,} in total discounts. Net sales were ${row[' Sales']}. "
                f"The Cost of Goods Sold (COGS) was ${row['COGS']:,}, resulting in a net profit of ${row['Profit']:,}."
            )
        
        doc = Document(
                page_content=row_text,
                metadata={
                        "source": "financial_data.xlsx",
                        "row_index": index,
                        "doc_type": "Excel",
                        "country": row['Country'],
                        "product": row['Product'],
                        "chunk_id": excel_chunk_ids[index]
                }
            )

        excel_chunks.append(doc)

    return excel_chunks
