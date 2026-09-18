from sentence_transformers import SentenceTransformer


class E5EmbeddingModel:
    """
    Multilingual E5 embedding model for semantic retrieval.
    """

    MODEL_NAME = "intfloat/multilingual-e5-base"

    def __init__(self):
        try:
            self.model = SentenceTransformer(self.MODEL_NAME, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, texts, show_progress_bar=False):
        """
        Convert one or more texts into embedding vectors.
        Uses document/passage formatting if a list is provided.
        """
        if isinstance(texts, str):
            return self.encode_query(texts, show_progress_bar=show_progress_bar)
        return self.encode_documents(texts, show_progress_bar=show_progress_bar)

    def encode_documents(self, texts, show_progress_bar=False):
        """
        Generate embeddings for stored website chunks.
        E5 expects the 'passage:' prefix for documents.
        """
        formatted = [
            text if text.startswith("passage: ") else f"passage: {text}"
            for text in texts
        ]

        return self.model.encode(
            formatted,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

    def encode_query(self, query, show_progress_bar=False):
        """
        Generate an embedding for a user search query.
        E5 expects the 'query:' prefix for queries.
        """
        formatted = query if query.startswith("query: ") else f"query: {query}"

        return self.model.encode(
            [formatted],
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )[0]

    @property
    def dimension(self):
        """
        Return embedding vector dimension.
        """
        return self.model.get_embedding_dimension()