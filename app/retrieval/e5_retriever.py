from app.embeddings.e5_embedder import E5EmbeddingModel
from app.vectorstore.e5_qdrant_store import E5QdrantStore


class E5Retriever:
    """
    Semantic retriever using multilingual-e5-base + Qdrant.
    """

    def __init__(self, top_k=5, embedder=None, store=None):
        self.embedder = embedder if embedder is not None else E5EmbeddingModel()
        self.store = store if store is not None else E5QdrantStore()
        self.top_k = top_k

    def search(self, query, top_k=None):
        limit = top_k if top_k is not None else self.top_k
        query_vector = self.embedder.encode_query(query)

        results = self.store.client.query_points(
            collection_name=self.store.COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        return results.points

    def close(self):
        self.store.close()


if __name__ == "__main__":
    retriever = E5Retriever(top_k=5)

    query = "What is the lithosphere?"

    results = retriever.search(query)

    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}\n")

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"ID: {result.id}")
        print(f"Score: {result.score:.4f}")
        print(f"Text: {result.payload['text'][:500]}")
        print()

    retriever.close()