import json

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.reranker import DeterministicReranker


CHUNKS_FILE = "data/chunks/natural_environment.json"


QUERIES = [
    "What is an ecosystem?",
    "How is climate different from weather?",
    "How do humans affect water?",
    "Where is most of Earth's water found?",
    "How much of Earth's surface is covered by ocean?",
    "How many major layers does Earth's atmosphere have?",
    "How does plate tectonics cause movement?",
    "Why is the atmosphere important?",
]


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    retriever = BM25Retriever(chunks)
    reranker = DeterministicReranker()

    print("=" * 70)
    print("QUERY-TYPE ROBUSTNESS TEST")
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
                f"Score {result['score']:.4f} | "
                f"BM25 {result['bm25_score']:.4f} | "
                f"Definition {result['definition_score']:.4f}"
            )

            print(f"Text: {chunk['text'][:180]}")


if __name__ == "__main__":
    main()