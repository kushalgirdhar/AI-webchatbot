from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class QdrantStore:
    """
    Local Qdrant vector database.
    """

    COLLECTION_NAME = "natural_environment"

    def __init__(self, path="qdrant_storage"):
        self.client = QdrantClient(path=path)

    def create_collection(self):
        collections = self.client.get_collections().collections

        existing_names = [collection.name for collection in collections]

        if self.COLLECTION_NAME not in existing_names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                ),
            )

            print(f"Collection created: {self.COLLECTION_NAME}")

        else:
            print(f"Collection already exists: {self.COLLECTION_NAME}")

    def close(self):
        self.client.close()

if __name__ == "__main__":
    store = QdrantStore()
    store.create_collection()
    store.close()