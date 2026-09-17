import json

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DeterministicReranker


CHUNKS_FILE = "data/chunks/natural_environment.json"


QUERIES = [
    "What is quantum computing?",
    "What is blockchain?",
    "What is machine learning?",
    "What is artificial intelligence?",
    "Who invented the telephone?",
]


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    retriever = BM25Retriever(chunks)
    reranker = DeterministicReranker()

    print("=" * 70)
    print("UNKNOWN / OUT-OF-SCOPE QUERY TEST")
    print("=" * 70)

    for query in QUERIES:
        candidates = retriever.search(query, top_k=20)
        results = reranker.rerank(query, candidates, top_k=5)

        print(f"\nQuery: {query}")

        for rank, result in enumerate(results, start=1):
            chunk = result["chunk"]

            print(
                f"Rank {rank} | "
                f"ID {chunk['id']} | "
                f"Score {result['score']:.4f}"
            )

            print(f"Text: {chunk['text'][:250]}")


if __name__ == "__main__":
    main()