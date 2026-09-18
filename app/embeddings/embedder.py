from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Local embedding model for semantic retrieval.
    """

    MODEL_NAME = "BAAI/bge-m3"

    def __init__(self):
        self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts):
        """
        Convert one or more texts into embedding vectors.
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    @property
    def dimension(self):
        """
        Return embedding vector dimension.
        """
        self.model.get_embedding_dimension()