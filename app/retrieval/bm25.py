from rank_bm25 import BM25Okapi

from app.retrieval.text_processor import tokenize


class BM25Retriever:
    """
    General-purpose BM25 retrieval engine.

    Works with any collection of text chunks.
    """

    def __init__(self, chunks):
        self.chunks = chunks

        # Build searchable text for every chunk
        self.searchable_texts = [self._build_search_text(chunk) for chunk in chunks]

        # Tokenize documents
        self.tokenized_documents = [tokenize(text) for text in self.searchable_texts]

        # Create BM25 index safely
        if self.tokenized_documents:
            self.bm25 = BM25Okapi(self.tokenized_documents)
        else:
            self.bm25 = None

    @staticmethod
    def _build_search_text(chunk):
        """
        Combine structural metadata with chunk content.
        """

        metadata = chunk.get("metadata", {})

        title = metadata.get("title", "")
        section = metadata.get("section", "")
        heading_path = metadata.get("heading_path", [])

        structural_text = " ".join(
            [
                title,
                section,
                section,
                " ".join(heading_path),
                " ".join(heading_path),
            ]
        )

        return f"{structural_text} {chunk['text']}"

    def search(self, query, top_k=5):
        """
        Search the indexed chunks and return the top K results.
        Returns an empty list for empty, invalid, or stop-word-only queries.
        """
        if not self.bm25 or not self.chunks:
            return []

        # Tokenize user query
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        # Calculate BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top K indexes
        top_indices = scores.argsort()[-top_k:][::-1]

        results = []
        for index in top_indices:
            results.append(
                {
                    "chunk": self.chunks[index],
                    "score": float(scores[index]),
                }
            )

        return results
