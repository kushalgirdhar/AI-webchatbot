import json

from app.retrieval.bm25 import BM25Retriever


CHUNKS_FILE = "data/chunks/natural_environment.json"
TEST_FILE = "data/evaluation/retrieval_generalization.json"


def evaluate(retriever, test_cases, k):
    successful = 0
    failures = []

    for test_case in test_cases:
        query = test_case["query"]
        expected_ids = set(test_case["expected_chunk_ids"])

        results = retriever.search(query, top_k=k)

        retrieved_ids = {
            result["chunk"]["id"]
            for result in results
        }

        if expected_ids.intersection(retrieved_ids):
            successful += 1
        else:
            failures.append(
                {
                    "query": query,
                    "expected": sorted(expected_ids),
                    "retrieved": sorted(retrieved_ids),
                }
            )

    recall = successful / len(test_cases)

    return recall, failures


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    retriever = BM25Retriever(chunks)

    print("=" * 70)
    print("BM25 CANDIDATE RECALL TEST")
    print("=" * 70)

    for k in [5, 10, 20, 30, 50, 61]:
        recall, failures = evaluate(
            retriever,
            test_cases,
            k,
        )

        print(f"\nRecall@{k}: {recall:.4f}")

        if failures:
            print(f"Failures: {len(failures)}")

            for failure in failures:
                print(f"  Query: {failure['query']}")
                print(f"  Expected: {failure['expected']}")
        else:
            print("Failures: 0")


if __name__ == "__main__":
    main()