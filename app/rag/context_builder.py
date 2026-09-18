class ContextBuilder:
    """
    Builds clean context from reranked retrieval results.
    """

    def __init__(self, max_chunks=3, max_chars=8000):
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def build(self, results):
        """
        Select the top useful chunks and combine them
        into a single context string.
        """

        if not results:
            return ""

        selected_chunks = results[:self.max_chunks]

        context_parts = []
        total_chars = 0

        for rank, result in enumerate(selected_chunks, start=1):
            chunk = result["chunk"]

            text = chunk.get("text", "").strip()
            metadata = chunk.get("metadata", {})

            if not text:
                continue

            title = metadata.get("title", "")
            section = metadata.get("section", "")
            heading_path = metadata.get("heading_path", [])

            heading = " > ".join(heading_path)

            source_info = []

            if title:
                source_info.append(f"Title: {title}")

            if section:
                source_info.append(f"Section: {section}")

            if heading:
                source_info.append(f"Heading: {heading}")

            source_block = "\n".join(source_info)

            block = (
                f"[Source {rank}]\n"
                f"{source_block}\n\n"
                f"{text}"
            )

            # Respect maximum context size
            remaining_chars = self.max_chars - total_chars

            if remaining_chars <= 0:
                break

            if len(block) > remaining_chars:
                block = block[:remaining_chars].rstrip()

            context_parts.append(block)
            total_chars += len(block)

        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    import json

    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.reranker import DeterministicReranker

    with open(
        "data/chunks/natural_environment.json",
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    query = "What is the lithosphere?"

    retriever = HybridRetriever(chunks, top_k=5)
    hybrid_results = retriever.search(query)

    reranker = DeterministicReranker()
    reranked_results = reranker.rerank(
        query,
        hybrid_results,
        top_k=5,
    )

    builder = ContextBuilder(
        max_chunks=3,
        max_chars=8000,
    )

    context = builder.build(reranked_results)

    print("=" * 60)
    print("FINAL CONTEXT")
    print("=" * 60)
    print(context)

    retriever.close()