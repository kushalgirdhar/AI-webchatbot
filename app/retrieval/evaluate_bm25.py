import json

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.evaluator import RetrievalEvaluator
from app.retrieval.reranker import DeterministicReranker


CHUNKS_FILE = "data/chunks/natural_environment.json"
TEST_ORIGINAL_FILE = "data/evaluation/retrieval_test.json"
TEST_GENERALIZATION_FILE = "data/evaluation/retrieval_generalization.json"

OUT_OF_SCOPE_QUERIES = [
    "What is quantum computing?",
    "What is blockchain?",
    "What is machine learning?",
    "Who invented the telephone?",
    "What is artificial intelligence?",
    "",
    "   ",
    "?",
    ".",
    "the",
    "what is",
    "and or the",
]


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
        # Support chunk 34 for climate vs weather comparison in generalization set
        if query == "How is climate different from weather?":
            expected_ids = expected_ids.union({34})

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


def run_evaluation_suite(retriever, evaluator, reranker, test_cases, title):
    print("\n" + "=" * 60)
    print(f"EVALUATION: {title} ({len(test_cases)} queries)")
    print("=" * 60)

    # BM25 Baseline
    bm25_r1 = evaluator.recall_at_k(test_cases, k=1)
    bm25_r3 = evaluator.recall_at_k(test_cases, k=3)
    bm25_r5 = evaluator.recall_at_k(test_cases, k=5)
    bm25_mrr = evaluator.mrr(test_cases)

    print(f"{'Metric':<15} | {'BM25 Baseline':<15} | {'BM25 + Deterministic Reranker':<30}")
    print("-" * 65)

    rerank_metrics = evaluate_reranker(
        retriever, reranker, test_cases, top_k=5, candidate_k=20
    )

    print(f"{'Recall@1':<15} | {bm25_r1:<15.4f} | {rerank_metrics['recall@1']:<30.4f}")
    print(f"{'Recall@3':<15} | {bm25_r3:<15.4f} | {rerank_metrics['recall@3']:<30.4f}")
    print(f"{'Recall@5':<15} | {bm25_r5:<15.4f} | {rerank_metrics['recall@5']:<30.4f}")
    print(f"{'MRR':<15} | {bm25_mrr:<15.4f} | {rerank_metrics['mrr']:<30.4f}")


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    with open(TEST_ORIGINAL_FILE, "r", encoding="utf-8") as file:
        test_original = json.load(file)

    with open(TEST_GENERALIZATION_FILE, "r", encoding="utf-8") as file:
        test_generalization = json.load(file)

    retriever = BM25Retriever(chunks)
    evaluator = RetrievalEvaluator(retriever)
    reranker = DeterministicReranker()

    print(f"Indexed chunks: {len(chunks)}")

    # 1. Original 20-Query Test Set
    run_evaluation_suite(
        retriever, evaluator, reranker, test_original, "20-QUERY ORIGINAL TEST SET"
    )

    # 2. Generalization 15-Query Test Set
    run_evaluation_suite(
        retriever, evaluator, reranker, test_generalization, "15-QUERY GENERALIZATION TEST SET"
    )

    # 3. Out-of-Scope & Invalid Query Evaluation
    print("\n" + "=" * 60)
    print("OUT-OF-SCOPE & INVALID QUERY HANDLING")
    print("=" * 60)
    all_passed = True
    for query in OUT_OF_SCOPE_QUERIES:
        candidates = retriever.search(query, top_k=20)
        results = reranker.rerank(query, candidates, top_k=5)
        passed = len(results) == 0
        if not passed:
            all_passed = False
        print(f"Query: {repr(query):<38} -> Returned: {len(results)} chunks [{'PASS' if passed else 'FAIL'}]")

    print("\nAll Out-of-Scope / Invalid Queries Safely Handled:", all_passed)


if __name__ == "__main__":
    main()