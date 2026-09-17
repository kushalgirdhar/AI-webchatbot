import json

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.text_processor import tokenize


CHUNKS_FILE = "data/chunks/natural_environment.json"


def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    retriever = BM25Retriever(chunks)

    query = "What is an ocean?"

    print("Query:")
    print(query)

    print("\nQuery tokens:")
    print(tokenize(query))

    results = retriever.search(
        query,
        top_k=5
    )

    print("\nTop 5 results:")

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print("\n" + "-" * 70)
        print(f"Rank: {rank}")
        print(f"ID: {chunk['id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Section: {chunk['metadata']['section']}")
        print(f"Heading path: {chunk['metadata']['heading_path']}")
        print(f"Text:\n{chunk['text']}")


if __name__ == "__main__":
    main()