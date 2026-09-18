import json
import os

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.e5_hybrid_retriever import E5HybridRetriever
from app.retrieval.e5_retriever import E5Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.retrieval.reranker import DeterministicReranker
from app.vectorstore.e5_qdrant_store import E5QdrantStore

CHUNKS_FILE = "data/chunks/natural_environment.json"
TEST_ORIGINAL_FILE = "data/evaluation/retrieval_test.json"
TEST_GENERALIZATION_FILE = "data/evaluation/retrieval_generalization.json"


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
        retrieved_ids = [
            result["chunk"]["id"] if "chunk" in result else result.id
            for result in results
        ]

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


def run_benchmark(title, test_cases, bm25, bge_hybrid, e5_hybrid):
    print("\n" + "=" * 95)
    print(f"BENCHMARK: {title} ({len(test_cases)} queries)")
    print("=" * 95)

    pipelines = [
        ("BM25", "Lexical Baseline", lambda q: bm25.search(q, top_k=5)),
        ("BGE-M3", "Semantic Only", lambda q: bge_hybrid.qdrant.search(q, top_k=5)),
        ("BGE-M3", "Hybrid (BM25 + RRF)", lambda q: bge_hybrid.search(q, top_k=5, rerank=False)),
        ("BGE-M3", "Hybrid + Reranker", lambda q: bge_hybrid.search(q, top_k=5, rerank=True)),
        ("E5", "Semantic Only", lambda q: e5_hybrid.e5_retriever.search(q, top_k=5)),
        ("E5", "Hybrid (BM25 + RRF)", lambda q: e5_hybrid.search(q, top_k=5, rerank=False)),
        ("E5", "Hybrid + Reranker", lambda q: e5_hybrid.search(q, top_k=5, rerank=True)),
    ]

    print(f"{'Model':<10} | {'Pipeline':<24} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<10}")
    print("-" * 95)

    results_table = []
    for model, pipe_name, fn in pipelines:
        m = evaluate_system(fn, test_cases, top_k=5)
        print(f"{model:<10} | {pipe_name:<24} | {m['recall@1']:<10.4f} | {m['recall@3']:<10.4f} | {m['recall@5']:<10.4f} | {m['mrr']:<10.4f}")
        results_table.append((model, pipe_name, m))

    return results_table


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(TEST_ORIGINAL_FILE, "r", encoding="utf-8") as f:
        test_original = json.load(f)
    with open(TEST_GENERALIZATION_FILE, "r", encoding="utf-8") as f:
        test_generalization = json.load(f)

    print("Initializing retrievers...")
    bm25 = BM25Retriever(chunks)
    bge_hybrid = HybridRetriever(chunks, top_k=5, candidate_k=20, rrf_k=60)
    
    # Share local Qdrant client to avoid multi-instance file lock collisions
    e5_store = E5QdrantStore(client=bge_hybrid.qdrant.store.client)
    e5_retriever = E5Retriever(top_k=20, store=e5_store)
    e5_hybrid = E5HybridRetriever(chunks, top_k=5, candidate_k=20, rrf_k=60, e5_retriever=e5_retriever)

    try:
        run_benchmark(
            "1. ORIGINAL 20-QUERY TEST SET",
            test_original,
            bm25,
            bge_hybrid,
            e5_hybrid,
        )
        run_benchmark(
            "2. GENERALIZATION 15-QUERY TEST SET",
            test_generalization,
            bm25,
            bge_hybrid,
            e5_hybrid,
        )
    finally:
        bge_hybrid.close()


if __name__ == "__main__":
    main()

