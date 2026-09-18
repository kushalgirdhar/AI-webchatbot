from app.embeddings.e5_embedder import E5EmbeddingModel


def main():
    embedder = E5EmbeddingModel()

    print("=" * 60)
    print("Multilingual E5 Embedding Test")
    print("=" * 60)

    print(f"Model: {embedder.MODEL_NAME}")
    print(f"Dimension: {embedder.dimension}")

    document = "The lithosphere is the rigid outer layer of Earth."
    query = "What is the lithosphere?"

    document_vector = embedder.encode_documents([document])[0]
    query_vector = embedder.encode_query(query)

    print(f"Document vector length: {len(document_vector)}")
    print(f"Query vector length: {len(query_vector)}")

    print("\nFirst 10 document values:")
    print(document_vector[:10])

    print("\nFirst 10 query values:")
    print(query_vector[:10])


if __name__ == "__main__":
    main()