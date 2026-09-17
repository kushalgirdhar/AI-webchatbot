import json

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.evaluator import RetrievalEvaluator
from app.retrieval.reranker import DeterministicReranker


CHUNKS_FILE = "data/chunks/natural_environment.json"
TEST_FILE = "data/evaluation/retrieval_generalization.json"

def evaluate_reranker(retriever, reranker, test_cases, top_k=5, candidate_k=20):
    """
    Evaluate two-stage retrieval:
    1. BM25 retrieves candidate_k candidates (high recall).
    2. Deterministic reranker re-orders candidates to return top_k results (high precision).
    """
    successful_at_1 = 0
    successful_at_3 = 0
    successful_at_k = 0
    reciprocal_ranks = []

    for test_case in test_cases:
        query = test_case["query"]
        expected_ids = set(test_case["expected_chunk_ids"])

        # Stage 1: Get BM25 candidates
        candidates = retriever.search(query, top_k=candidate_k)

        # Stage 2: Rerank candidates
        results = reranker.rerank(query, candidates, top_k=top_k)

        retrieved_ids = [result["chunk"]["id"] for result in results]

        # Calculate Recall@1, Recall@3, Recall@K
        if expected_ids.intersection(retrieved_ids[:1]):
            successful_at_1 += 1

        if expected_ids.intersection(retrieved_ids[:3]):
            successful_at_3 += 1

        if expected_ids.intersection(retrieved_ids):
            successful_at_k += 1

        # Calculate Reciprocal Rank
        reciprocal_rank = 0.0
        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk"]["id"]
            if chunk_id in expected_ids:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    recall_1 = successful_at_1 / len(test_cases)
    recall_3 = successful_at_3 / len(test_cases)
    recall_k = successful_at_k / len(test_cases)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {
        "recall@1": recall_1,
        "recall@3": recall_3,
        f"recall@{top_k}": recall_k,
        "mrr": mrr,
    }


def main():
    # Load chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Load evaluation test cases
    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    print(f"Total chunks: {len(chunks)}")
    print(f"Test cases: {len(test_cases)}")

    # Create BM25 retriever
    retriever = BM25Retriever(chunks)

    # Create evaluator
    evaluator = RetrievalEvaluator(retriever)

    # Create reranker
    reranker = DeterministicReranker()

    # --------------------------------------------------
    # BM25 BASELINE
    # --------------------------------------------------
    bm25_recall_at_1 = evaluator.recall_at_k(test_cases, k=1)
    bm25_recall_at_3 = evaluator.recall_at_k(test_cases, k=3)
    bm25_recall_at_5 = evaluator.recall_at_k(test_cases, k=5)
    bm25_mrr = evaluator.mrr(test_cases)

    print("\n" + "=" * 50)
    print("BM25 BASELINE EVALUATION")
    print("=" * 50)
    print(f"Recall@1: {bm25_recall_at_1:.4f}")
    print(f"Recall@3: {bm25_recall_at_3:.4f}")
    print(f"Recall@5: {bm25_recall_at_5:.4f}")
    print(f"MRR:      {bm25_mrr:.4f}")

    # --------------------------------------------------
    # BM25 + DETERMINISTIC RERANKER
    # --------------------------------------------------
    rerank_metrics = evaluate_reranker(
        retriever, reranker, test_cases, top_k=5, candidate_k=20
    )

    print("\n" + "=" * 50)
    print("BM25 + DETERMINISTIC RERANKER (Two-Stage)")
    print("=" * 50)
    print(f"Recall@1: {rerank_metrics['recall@1']:.4f}")
    print(f"Recall@3: {rerank_metrics['recall@3']:.4f}")
    print(f"Recall@5: {rerank_metrics['recall@5']:.4f}")
    print(f"MRR:      {rerank_metrics['mrr']:.4f}")


if __name__ == "__main__":
    main()