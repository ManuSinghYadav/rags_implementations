## RAG 1

- Based on ++single source++ i.e. Docling's paper
- Implemented RAG pipeline (++Parsing++ -> ++Chunking++ -> ++Embedding++ -> ++RAG++)
- Implemented Evals (++Created golden dataset by a LLM++ -> ++Used my own package for retrieval accuracy++ -> ++Used DeepEval for generation (+retrieval) accuracy++)

#### Level up from here

- Introduce multiple documents and do chunking and embedding
- Improve chunking and embedding through retrieval evals (yes, before touching LLM) (log them) (see [here](https://gemini.google.com/app/0f17315361acdc8f))
- Use of muliple chunks in retrieval and golden dataset
- Understanding of each evaluation matric
- Improve generation at the last (log them)

