import json

from qdrant_client.models import PointStruct

from app.embeddings.e5_embedder import E5EmbeddingModel
from app.vectorstore.e5_qdrant_store import E5QdrantStore


CHUNKS_FILE = "data/chunks/natural_environment.json"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    print("=" * 70)
    print("E5 → Qdrant Indexing")
    print("=" * 70)

    chunks = load_chunks()
    print(f"Loaded chunks: {len(chunks)}")

    embedder = E5EmbeddingModel()

    print(f"Model: {embedder.MODEL_NAME}")
    print(f"Vector dimension: {embedder.dimension}")

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating E5 embeddings...")
    vectors = embedder.encode_documents(texts, show_progress_bar=True)

    print(f"Generated vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    store = E5QdrantStore()
    store.create_collection()

    points = []

    for chunk, vector in zip(chunks, vectors):
        point = PointStruct(
            id=chunk["id"],
            vector=vector.tolist(),
            payload={
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            },
        )

        points.append(point)

    print("\nUploading vectors to Qdrant...")

    store.client.upsert(
        collection_name=store.COLLECTION_NAME,
        points=points,
    )

    print(f"Inserted points: {len(points)}")
    print(f"Collection: {store.COLLECTION_NAME}")

    store.close()

    print("\n" + "=" * 70)
    print("E5 indexing completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()