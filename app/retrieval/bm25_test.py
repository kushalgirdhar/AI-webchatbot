import json

from app.retrieval.bm25 import BM25Retriever


INPUT_FILE = "data/chunks/natural_environment.json"

QUERIES = [
    "What are the components of the natural environment?",
    "What is climate?",
    "What are biogeochemical cycles?",
    "How does human activity affect the environment?",
    "What is wilderness?",
]


def main():
    # Load chunks
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    print(f"Total chunks: {len(chunks)}")

    # Create reusable BM25 retriever
    retriever = BM25Retriever(chunks)

    # Test multiple queries
    for query_number, query in enumerate(QUERIES, start=1):

        results = retriever.search(
            query,
            top_k=5
        )

        print("\n" + "=" * 70)
        print(f"QUERY {query_number}")
        print("=" * 70)
        print(query)

        print("\nTop 5 results:\n")

        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]
            score = result["score"]

            print(f"--- Rank {rank} ---")
            print(f"Score: {score:.4f}")
            print(
                f"Section: "
                f"{chunk['metadata']['section']}"
            )
            print(
                f"Text: "
                f"{chunk['text'][:300]}"
            )
            print()


if __name__ == "__main__":
    main()