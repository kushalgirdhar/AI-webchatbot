from qdrant_client import QdrantClient


client = QdrantClient(url="http://localhost:6333")

collection_name = "natural_environment"


points, next_page = client.scroll(
    collection_name=collection_name,
    limit=100,
    with_payload=True,
    with_vectors=False,
)


points = sorted(points, key=lambda point: point.id)


print()
print("=" * 80)
print("QDRANT COLLECTION")
print("=" * 80)
print(f"Collection : {collection_name}")
print(f"Total chunks: {len(points)}")
print("=" * 80)


for point in points:

    payload = point.payload or {}
    text = payload.get("text", "")
    metadata = payload.get("metadata", {})

    print()
    print("=" * 80)
    print(f"CHUNK ID: {point.id}")
    print("=" * 80)

    print("\nTEXT")
    print("-" * 80)
    print(text)

    print("\nMETADATA")
    print("-" * 80)

    print(f"Title       : {metadata.get('title', '')}")
    print(f"Section     : {metadata.get('section', '')}")

    heading_path = metadata.get("heading_path", [])

    if heading_path:
        print(f"Heading Path: {' > '.join(heading_path)}")
    else:
        print("Heading Path: ")

print()
print("=" * 80)
print("END OF QDRANT DATA")
print("=" * 80)


client.close()