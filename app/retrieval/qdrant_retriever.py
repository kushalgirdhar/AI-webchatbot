from app.embeddings.embedder import EmbeddingModel
from app.vectorstore.qdrant_store import QdrantStore


class QdrantRetriever:
    """
    Semantic retriever using BGE-M3 + Qdrant.
    """

    def __init__(self, top_k=5):
        self.embedder = EmbeddingModel()
        self.store = QdrantStore()
        self.top_k = top_k

    def search(self, query, top_k=None):
        limit = top_k if top_k is not None else self.top_k

        # Convert query into a vector
        query_vector = self.embedder.encode([query])[0]

        # Search Qdrant
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
    retriever = QdrantRetriever(top_k=5)

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