import json

from app.embeddings.embedder import EmbeddingModel
from app.vectorstore.qdrant_store import QdrantStore
from qdrant_client.models import PointStruct


CHUNKS_FILE = "data/chunks/natural_environment.json"


def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    # Load chunks
    chunks = load_chunks()

    print(f"Loaded chunks: {len(chunks)}")

    # Load embedding model
    embedder = EmbeddingModel()

    # Prepare texts
    texts = [chunk["text"] for chunk in chunks]

    # Generate embeddings
    print("Generating embeddings...")

    vectors = embedder.encode(texts)

    print(f"Generated vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    # Connect to Qdrant
    store = QdrantStore()

    # Create Qdrant points
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

    # Insert points into Qdrant
    store.client.upsert(
        collection_name=store.COLLECTION_NAME,
        points=points,
    )

    print(f"Inserted points into Qdrant: {len(points)}")

    # Close connection
    store.close()


if __name__ == "__main__":
    main()