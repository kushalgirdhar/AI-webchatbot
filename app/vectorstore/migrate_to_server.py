from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


COLLECTION_NAME = "natural_environment"


# Existing local Qdrant database
local_client = QdrantClient(path="qdrant_storage")

# Running Qdrant server
server_client = QdrantClient(
    url="http://localhost:6333"
)


# Get collection information
collection_info = local_client.get_collection(COLLECTION_NAME)

vector_size = collection_info.config.params.vectors.size
distance = collection_info.config.params.vectors.distance

print(f"Collection: {COLLECTION_NAME}")
print(f"Vector size: {vector_size}")
print(f"Distance: {distance}")


# Create collection on Qdrant server
if COLLECTION_NAME not in [
    collection.name
    for collection in server_client.get_collections().collections
]:
    server_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=distance,
        ),
    )

    print("Collection created on Qdrant server.")
else:
    print("Collection already exists on Qdrant server.")


# Read all points from local database
points, _ = local_client.scroll(
    collection_name=COLLECTION_NAME,
    limit=100,
    with_payload=True,
    with_vectors=True,
)

print(f"Points read from local database: {len(points)}")


# Convert points for server
server_points = []

for point in points:
    server_points.append(
        PointStruct(
            id=point.id,
            vector=point.vector,
            payload=point.payload,
        )
    )


# Upload points to server
server_client.upsert(
    collection_name=COLLECTION_NAME,
    points=server_points,
)

print(f"Points uploaded to Qdrant server: {len(server_points)}")


# Close local client
local_client.close()

print("Migration completed.")