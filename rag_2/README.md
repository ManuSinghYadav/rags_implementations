### RAG 2

- Multiple data source (PDF: Apple_10k_Form, Excel: Sample_Financial_Data)
- We're parsing excel sheet as "the Row-as-a-String Method" *(drawback: not able to answe aggreation queries)*

##### Flow

1. So first we'll do parsing by PyMuPDF4LLM, we'll check the retrieval matrics, and then check by docling.

##### Level up from here

1. Introduce **router**, *"If the question is about text, send it to the PDF vector database. If the question involves numbers/math, send it to the Excel SQL database."*
2. **Text-to-sql:** Instead of row-as-a-string method for excel and storing it in vdb, we'll store in local SQLite, and LLM will write SQL queries to retrieve the data.

