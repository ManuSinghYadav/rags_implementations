def reciprocal_rank_fusion(
    vector_results: list, bm25_results: list, k: int = 60, top_n: int = 10
) -> list:
    rrf_scores = {}

    def score_results(doc_list):
        for rank, doc in enumerate(doc_list, start=1):
            # Extract the unique string ID to use as a hashable key
            chunk_id = doc.metadata["chunk_id"]

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"doc": doc, "score": 0.0}

            rrf_scores[chunk_id]["score"] += 1.0 / (k + rank)

    score_results(vector_results)
    score_results(bm25_results)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

    return [item["doc"] for item in sorted_items[:top_n]]
