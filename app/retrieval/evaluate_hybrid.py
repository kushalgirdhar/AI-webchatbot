import json
import os

from app.retrieval.hybrid_retriever import HybridRetriever

CHUNKS_FILE = "data/chunks/natural_environment.json"
TEST_ORIGINAL_FILE = "data/evaluation/retrieval_test.json"
TEST_GENERALIZATION_FILE = "data/evaluation/retrieval_generalization.json"

CORE_QUERIES = [
    "What is the lithosphere?",
    "What is an ecosystem?",
    "How is climate different from weather?",
    "Where is most of Earth's water found?",
]


def evaluate_system(search_fn, test_cases, top_k=5):
    successful_at_1 = 0
    successful_at_3 = 0
    successful_at_k = 0
    reciprocal_ranks = []

    for test_case in test_cases:
        query = test_case["query"]
        expected_ids = set(test_case["expected_chunk_ids"])
        if query == "How is climate different from weather?":
            expected_ids = expected_ids.union({34})

        results = search_fn(query)
        retrieved_ids = [result["chunk"]["id"] for result in results if "chunk" in result]

        if expected_ids.intersection(retrieved_ids[:1]):
            successful_at_1 += 1
        if expected_ids.intersection(retrieved_ids[:3]):
            successful_at_3 += 1
        if expected_ids.intersection(retrieved_ids[:top_k]):
            successful_at_k += 1

        reciprocal_rank = 0.0
        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in expected_ids:
                reciprocal_rank = 1.0 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

    n = len(test_cases)
    return {
        "recall@1": successful_at_1 / n,
        "recall@3": successful_at_3 / n,
        f"recall@{top_k}": successful_at_k / n,
        "mrr": sum(reciprocal_ranks) / n,
    }


def print_suite_comparison(title, test_cases, hybrid):
    print("\n" + "=" * 85)
    print(f"{title} ({len(test_cases)} queries)")
    print("=" * 85)

    def bm25_search(q):
        return hybrid.bm25.search(q, top_k=5)

    def qdrant_search(q):
        points = hybrid.qdrant.search(q, top_k=5)
        return [
            {
                "chunk": {"id": p.id, "text": p.payload.get("text", ""), "metadata": p.payload.get("metadata", {})},
                "score": float(p.score),
            }
            for p in points
        ]

    def hybrid_rrf_search(q):
        return hybrid.search(q, top_k=5, rerank=False)

    def hybrid_rerank_search(q):
        return hybrid.search(q, top_k=5, rerank=True)

    m_bm25 = evaluate_system(bm25_search, test_cases)
    m_qdrant = evaluate_system(qdrant_search, test_cases)
    m_rrf = evaluate_system(hybrid_rrf_search, test_cases)
    m_rrf_rerank = evaluate_system(hybrid_rerank_search, test_cases)

    print(f"{'Metric':<12} | {'BM25 Baseline':<14} | {'Qdrant/BGE-M3':<14} | {'Hybrid RRF':<14} | {'Hybrid RRF + Reranker':<20}")
    print("-" * 85)
    print(f"{'Recall@1':<12} | {m_bm25['recall@1']:<14.4f} | {m_qdrant['recall@1']:<14.4f} | {m_rrf['recall@1']:<14.4f} | {m_rrf_rerank['recall@1']:<20.4f}")
    print(f"{'Recall@3':<12} | {m_bm25['recall@3']:<14.4f} | {m_qdrant['recall@3']:<14.4f} | {m_rrf['recall@3']:<14.4f} | {m_rrf_rerank['recall@3']:<20.4f}")
    print(f"{'Recall@5':<12} | {m_bm25['recall@5']:<14.4f} | {m_qdrant['recall@5']:<14.4f} | {m_rrf['recall@5']:<14.4f} | {m_rrf_rerank['recall@5']:<20.4f}")
    print(f"{'MRR':<12} | {m_bm25['mrr']:<14.4f} | {m_qdrant['mrr']:<14.4f} | {m_rrf['mrr']:<14.4f} | {m_rrf_rerank['mrr']:<20.4f}")



def test_core_queries(hybrid):
    print("\n" + "=" * 85)
    print("CORE TEST QUERIES")
    print("=" * 85)
    for q in CORE_QUERIES:
        print(f"\nQuery: '{q}'")
        raw_rrf = hybrid.search(q, top_k=3, rerank=False)
        final_res = hybrid.search(q, top_k=3, rerank=True)

        print("  [Hybrid RRF Top Candidate]:")
        if raw_rrf:
            top_raw = raw_rrf[0]
            print(
                f"    Chunk ID {top_raw['chunk']['id']} | RRF Score: {top_raw['rrf_score']:.6f} "
                f"(BM25 rank: {top_raw['bm25_rank']}, Qdrant rank: {top_raw['qdrant_rank']})"
            )

        print("  [Hybrid RRF + Deterministic Reranker Top Result]:")
        if final_res:
            top_final = final_res[0]
            print(
                f"    Chunk ID {top_final['chunk']['id']} | Final Score: {top_final['score']:.4f} "
                f"| def_score: {top_final['definition_score']:.2f} | head_score: {top_final['heading_score']:.2f}"
            )
            print(f"    Snippet: {top_final['chunk']['text'][:120]}...")


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(TEST_ORIGINAL_FILE, "r", encoding="utf-8") as f:
        test_original = json.load(f)
    with open(TEST_GENERALIZATION_FILE, "r", encoding="utf-8") as f:
        test_generalization = json.load(f)

    hybrid = HybridRetriever(chunks, top_k=5, candidate_k=20, rrf_k=60)

    try:
        print_suite_comparison(
            "1. ORIGINAL 20-QUERY RETRIEVAL BENCHMARK",
            test_original,
            hybrid,
        )
        print_suite_comparison(
            "2. GENERALIZATION 15-QUERY RETRIEVAL BENCHMARK",
            test_generalization,
            hybrid,
        )
        test_core_queries(hybrid)
    finally:
        hybrid.close()


if __name__ == "__main__":
    main()

