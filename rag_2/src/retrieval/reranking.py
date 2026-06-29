import yaml

from rag_2.src.config import CONFIG
from rag_2.src.core.clients import co


def cohere_reranking(question: str, retrieved_chunks: list) -> list:

	# Creating a YAML format, as we have metatdata as well (https://docs.cohere.com/docs/rerank-overview#example-with-structured-data)
	docs = [{
	'Header 1': i.metadata.get('Header 1'), 
	'Header 2': i.metadata.get('Header 2'),
	'content': i.page_content,
	} for i in retrieved_chunks]

	yaml_docs = [yaml.dump(doc, sort_keys=False) for doc in docs]

	results = co.rerank(
		model="rerank-v4.0-pro",
		query=question,
		documents=yaml_docs,
		top_n=CONFIG['cohere_k'],
		)
	
	return [retrieved_chunks[i.index] for i in results.results]
	