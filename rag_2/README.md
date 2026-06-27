### RAG 2

- Multiple data source (PDF: Apple_10k_Form, Excel: Sample_Financial_Data)
- We're parsing excel sheet as "the Row-as-a-String Method" *(drawback: not able to answe aggreation queries)*

##### Flow

1. So first we'll do parsing by PyMuPDF4LLM, we'll check the retrieval matrics, and then check by docling.

##### Level up from here

1. Use **multiple types of data sources**, one book, one research paper, one codebase, one excel.
2. Use **router**, to see what type of question is it.
3. **Different chunking strategies** for all: Reseach paper will have **markdown chunking**, book will have **markdown + sementic chunking**, excel will have both row as a string method and **text to sql**.
4. Explore **metadata tagging**

##### Findings

1. For excel data, **row-as-a-string** method is extremely bad for multipe chunks, text-to-sql wins.
