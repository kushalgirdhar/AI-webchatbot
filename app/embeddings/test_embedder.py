from app.embeddings.embedder import EmbeddingModel


def main():
    embedder = EmbeddingModel()

    text = "The lithosphere is the rigid outer layer of Earth."

    vector = embedder.encode([text])[0]

    print("Model:", embedder.MODEL_NAME)
    print("Dimension:", embedder.dimension)
    print("Vector length:", len(vector))
    print("First 10 values:", vector[:10])


if __name__ == "__main__":
    main()