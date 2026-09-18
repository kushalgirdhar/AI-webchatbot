import json

from app.retrieval.e5_retriever import E5Retriever


TEST_FILE = "data/evaluation/retrieval_test.json"
GENERALIZATION_FILE = "data/evaluation/retrieval_generalization.json"


def load_evaluation_file(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate(retriever, evaluation_data, top_k=5):
    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []

    for item in evaluation_data:
        query = item["query"]
        expected_ids = set(item["expected_chunk_ids"])

        results = retriever.search(query)
        retrieved_ids = [result.id for result in results]

        # Recall@1
        if any(chunk_id in expected_ids for chunk_id in retrieved_ids[:1]):
            recall_at_1 += 1

        # Recall@3
        if any(chunk_id in expected_ids for chunk_id in retrieved_ids[:3]):
            recall_at_3 += 1

        # Recall@5
        if any(chunk_id in expected_ids for chunk_id in retrieved_ids[:5]):
            recall_at_5 += 1

        # MRR
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in expected_ids:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    total = len(evaluation_data)

    return {
        "Recall@1": recall_at_1 / total,
        "Recall@3": recall_at_3 / total,
        "Recall@5": recall_at_5 / total,
        "MRR": sum(reciprocal_ranks) / total,
    }


def print_results(name, results):
    print(f"\n{name}")
    print("-" * 40)

    for metric, value in results.items():
        print(f"{metric:<12}: {value:.4f}")


def main():
    print("=" * 70)
    print("Multilingual-E5 Retrieval Evaluation")
    print("=" * 70)

    retriever = E5Retriever(top_k=5)

    test_data = load_evaluation_file(TEST_FILE)
    generalization_data = load_evaluation_file(GENERALIZATION_FILE)

    print(f"\nStandard test queries: {len(test_data)}")
    print(f"Generalization queries: {len(generalization_data)}")

    test_results = evaluate(
        retriever,
        test_data,
    )

    generalization_results = evaluate(
        retriever,
        generalization_data,
    )

    print_results(
        "Standard Retrieval Test",
        test_results,
    )

    print_results(
        "Generalization Test",
        generalization_results,
    )

    retriever.close()

    print("\n" + "=" * 70)
    print("Evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()