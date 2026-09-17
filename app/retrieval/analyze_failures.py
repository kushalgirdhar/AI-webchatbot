import json

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DeterministicReranker


CHUNKS_FILE = "data/chunks/natural_environment.json"
TEST_FILE = "data/evaluation/retrieval_generalization.json"


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    with open(TEST_FILE, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    retriever = BM25Retriever(chunks)
    reranker = DeterministicReranker()

    print("=" * 70)
    print("RERANKER RANK-1 FAILURE ANALYSIS")
    print("=" * 70)

    failures = 0

    for test_case in test_cases:
        query = test_case["query"]
        expected_ids = set(test_case["expected_chunk_ids"])

        candidates = retriever.search(query, top_k=20)
        results = reranker.rerank(query, candidates, top_k=5)

        top_id = results[0]["chunk"]["id"]

        if top_id not in expected_ids:
            failures += 1

            print(f"\nQuery: {query}")
            print(f"Expected chunk IDs: {sorted(expected_ids)}")
            print(f"Top result: {top_id}")

            print("\nTop 5 results:")
            for rank, result in enumerate(results, start=1):
                chunk = result["chunk"]

                print(
                    f"\nRank {rank} | "
                    f"ID {chunk['id']} | "
                    f"Final score: {result['score']:.4f} | "
                    f"BM25: {result['bm25_score']:.4f}"
                )

                print(
                    f"Coverage: {result['term_coverage']:.4f} | "
                    f"Phrase: {result['exact_phrase_score']:.4f} | "
                    f"Definition: {result['definition_score']:.4f} | "
                    f"Heading: {result['heading_score']:.4f} | "
                    f"Lead: {result['lead_score']:.4f}"
                )

                print("Text:")
                print(chunk["text"][:500])

            print("-" * 70)

    print(f"\nTotal rank-1 failures: {failures}")
    

if __name__ == "__main__":
    main()