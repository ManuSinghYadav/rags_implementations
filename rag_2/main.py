from rag_2.src.core.clients import openai
from rag_2.src.retrieval.reranking import cohere_reranking
from rag_2.src.retrieval.vdb_retrieval import retrieve_filtered_vector_docs
from rag_2.src.retrieval.metadata_filtering import extract_query_intent
from rag_2.src.retrieval.bm25s_retrieval import bm25_retriever_search
from rag_2.src.retrieval.hybrid_search import reciprocal_rank_fusion


def context_building(context: str) -> str:
	extended_prompt = f"""You are an smart AI assistant helps in reading form 10k and financial data. You will answer the question in explainatory way.
	If relevant, use the given context to answer any question.
	If you don't know the answer, say so.
	Context:
	{context}"""
	return extended_prompt

def rag_pipeline(query: str) -> str:
	has_any_filter = extract_query_intent(query)
	cleaned_query = has_any_filter.cleaned_query
	file_filter_str = has_any_filter.file_type if has_any_filter.has_filter else None

	vdb_retrieved_docs = retrieve_filtered_vector_docs(cleaned_query, has_any_filter)
	bm25s_results = bm25_retriever_search(cleaned_query, file_type_filter=file_filter_str)
	hybrid_retrieval = reciprocal_rank_fusion(bm25s_results, vdb_retrieved_docs)

	reranking = cohere_reranking(cleaned_query, hybrid_retrieval)
	context = ['\n\n' + i.page_content for i in reranking]
	extended_prompt = context_building(context)
	result = openai.chat.completions.create(
		model="gpt-4o-mini",
		messages=[{'role': 'system', 'content': extended_prompt}, {'role': 'user', 'content': cleaned_query}]
	)
	result = result.choices[0].message.content
	data = {
		"question": cleaned_query,
		"generated_answer": result,
		"contexts": [i.page_content for i in reranking],
		"chunk_used": [i.metadata['chunk_id'] for i in reranking]
	}
	print(data)
	return data

if __name__ == '__main__':
    rag_pipeline("What were the total government sales in Canada for January?")