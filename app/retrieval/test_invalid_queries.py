import json

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DeterministicReranker


CHUNKS_FILE = "data/chunks/natural_environment.json"

QUERIES = [
    "",
    " ",
    "?",
    ".",
    "the",
    "what is",
    "and or the",
]


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    retriever = BM25Retriever(chunks)
    reranker = DeterministicReranker()

    print("=" * 70)
    print("EMPTY / INVALID QUERY TEST")
    print("=" * 70)

    for query in QUERIES:
        print(f"\nQuery: {repr(query)}")

        try:
            candidates = retriever.search(query, top_k=20)
            results = reranker.rerank(query, candidates, top_k=5)

            if not results:
                print("Result: No results")
                continue

            for rank, result in enumerate(results, start=1):
                chunk = result["chunk"]
                print(
                    f"Rank {rank} | "
                    f"ID {chunk['id']} | "
                    f"Score {result['score']:.4f}"
                )

        except Exception as error:
            print(f"ERROR: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()