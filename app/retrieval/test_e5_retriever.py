from app.retrieval.e5_retriever import E5Retriever


def main():
    retriever = E5Retriever(top_k=5)

    query = "What is the lithosphere?"

    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)

    results = retriever.search(query)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print(f"Chunk ID: {result.id}")
        print(f"Score: {result.score:.4f}")

        payload = result.payload or {}
        print(f"Text: {payload.get('text', '')}")

    retriever.close()


if __name__ == "__main__":
    main()