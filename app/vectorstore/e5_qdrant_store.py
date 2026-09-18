from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class E5QdrantStore:
    """
    Qdrant storage for multilingual-e5-base embeddings.
    """

    DEFAULT_COLLECTION_NAME = "natural_environment_e5"
    VECTOR_SIZE = 768

    def __init__(self, path="qdrant_storage", url=None, client=None, collection_name=None):
        if client is not None:
            self.client = client
        elif url is not None:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=path)
        self.COLLECTION_NAME = collection_name or self.DEFAULT_COLLECTION_NAME

    def create_collection(self, collection_name=None):
        target_name = collection_name or self.COLLECTION_NAME
        collections = self.client.get_collections().collections
        existing_names = [collection.name for collection in collections]

        if target_name not in existing_names:
            self.client.create_collection(
                collection_name=target_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Collection created: {target_name}")
        else:
            print(f"Collection already exists: {target_name}")

    def close(self):
        self.client.close()


if __name__ == "__main__":
    store = E5QdrantStore()
    store.create_collection()
    store.close()