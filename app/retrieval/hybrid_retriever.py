from app.retrieval.bm25 import BM25Retriever
from app.retrieval.qdrant_retriever import QdrantRetriever
from app.retrieval.reranker import DeterministicReranker


class HybridRetriever:
    """
    Combines BM25 keyword retrieval with BGE-M3 semantic retrieval from Qdrant
    using Reciprocal Rank Fusion (RRF) and a deterministic reranker.
    """

    def __init__(
        self,
        chunks,
        top_k=5,
        candidate_k=20,
        rrf_k=60,
        bm25=None,
        qdrant=None,
        reranker=None,
    ):
        self.chunks = chunks
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

        self.bm25 = bm25 if bm25 is not None else BM25Retriever(chunks)
        self.qdrant = qdrant if qdrant is not None else QdrantRetriever(top_k=self.candidate_k)
        self.reranker = reranker if reranker is not None else DeterministicReranker()

    def fuse_rrf(self, bm25_results, qdrant_results, rrf_k=None):
        """
        Fuse BM25 and Qdrant candidate rankings using Reciprocal Rank Fusion (RRF):
        RRF(d) = sum_{m in M} 1 / (k + rank_m(d))
        """
        k = rrf_k if rrf_k is not None else self.rrf_k
        candidates = {}

        # 1. Process BM25 candidates
        for rank, result in enumerate(bm25_results, start=1):
            chunk = result["chunk"]
            chunk_id = chunk["id"]

            candidates[chunk_id] = {
                "chunk": chunk,
                "bm25_score": result.get("score", 0.0),
                "qdrant_score": 0.0,
                "bm25_rank": rank,
                "qdrant_rank": None,
                "rrf_score": 1.0 / (k + rank),
            }

        # 2. Process Qdrant semantic candidates
        for rank, result in enumerate(qdrant_results, start=1):
            chunk_id = result.id
            score = float(result.score)
            payload = result.payload

            if chunk_id not in candidates:
                chunk = {
                    "id": chunk_id,
                    "text": payload.get("text", ""),
                    "metadata": payload.get("metadata", {}),
                }
                candidates[chunk_id] = {
                    "chunk": chunk,
                    "bm25_score": 0.0,
                    "qdrant_score": score,
                    "bm25_rank": None,
                    "qdrant_rank": rank,
                    "rrf_score": 1.0 / (k + rank),
                }
            else:
                candidates[chunk_id]["qdrant_rank"] = rank
                candidates[chunk_id]["qdrant_score"] = score
                candidates[chunk_id]["rrf_score"] += 1.0 / (k + rank)

        # Sort candidates by fused RRF score descending
        fused = sorted(
            candidates.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )
        return fused

    def search_candidates(self, query, candidate_k=None, rrf_k=None):
        """
        Retrieve raw fused candidates using RRF without the final deterministic reranker.
        """
        k_candidates = candidate_k if candidate_k is not None else self.candidate_k
        bm25_results = self.bm25.search(query, top_k=k_candidates)
        qdrant_results = self.qdrant.search(query, top_k=k_candidates)

        fused = self.fuse_rrf(bm25_results, qdrant_results, rrf_k=rrf_k)
        return fused

    def search(
        self,
        query,
        top_k=None,
        candidate_k=None,
        rrf_k=None,
        rerank=True,
    ):
        """
        Perform hybrid retrieval:
        1. Retrieve top candidates from BM25 and Qdrant.
        2. Combine candidates using Reciprocal Rank Fusion (RRF).
        3. Pass hybrid candidates to the Deterministic Reranker (if rerank=True).
        """
        limit = top_k if top_k is not None else self.top_k
        fused_candidates = self.search_candidates(
            query,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
        )

        if not fused_candidates:
            return []

        if not rerank:
            return fused_candidates[:limit]

        # Prepare candidates for deterministic reranker
        rerank_inputs = []
        for candidate in fused_candidates:
            rerank_inputs.append(
                {
                    "chunk": candidate["chunk"],
                    "score": candidate["bm25_score"],
                    "rrf_score": candidate["rrf_score"],
                    "bm25_score": candidate["bm25_score"],
                    "qdrant_score": candidate["qdrant_score"],
                    "bm25_rank": candidate["bm25_rank"],
                    "qdrant_rank": candidate["qdrant_rank"],
                }
            )

        reranked = self.reranker.rerank(query, rerank_inputs, top_k=limit)
        return reranked

    def close(self):
        self.qdrant.close()


if __name__ == "__main__":
    import json

    with open(
        "data/chunks/natural_environment.json",
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    retriever = HybridRetriever(chunks, top_k=5)

    test_queries = [
        "What is the lithosphere?",
        "What is an ecosystem?",
        "How is climate different from weather?",
        "Where is most of Earth's water found?",
    ]

    for query in test_queries:
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        # 1. Raw RRF Candidates
        rrf_candidates = retriever.search(query, top_k=5, rerank=False)
        print("\n--- Top 5 Hybrid RRF Candidates ---")
        for rank, res in enumerate(rrf_candidates, start=1):
            c = res["chunk"]
            print(
                f"Rank {rank} | Chunk ID: {c['id']} | RRF: {res['rrf_score']:.6f} "
                f"| BM25 rank: {res['bm25_rank']} (score: {res['bm25_score']:.4f}) "
                f"| Qdrant rank: {res['qdrant_rank']} (score: {res['qdrant_score']:.4f})"
            )

        # 2. Final Reranked Results
        reranked_results = retriever.search(query, top_k=5, rerank=True)
        print("\n--- Top 5 Hybrid RRF + Deterministic Reranker ---")
        for rank, res in enumerate(reranked_results, start=1):
            c = res["chunk"]
            print(
                f"Rank {rank} | Chunk ID: {c['id']} | Final Score: {res['score']:.4f} "
                f"| def_score: {res['definition_score']:.2f} | head_score: {res['heading_score']:.2f}"
            )
            print(f"Text snippet: {c['text'][:180]}...")
            print()

    retriever.close()