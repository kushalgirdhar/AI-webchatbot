class RetrievalEvaluator:
    """
    Evaluate a retrieval system using queries and expected chunk IDs.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def recall_at_k(self, test_cases, k=5):
        """
        Calculate Recall@K.

        A test case is considered successful when at least one
        expected chunk is present in the top K retrieved results.
        """

        if not test_cases:
            return 0.0

        successful = 0

        for test_case in test_cases:
            query = test_case["query"]
            expected_ids = set(test_case["expected_chunk_ids"])

            results = self.retriever.search(query, top_k=k)

            retrieved_ids = {
                result["chunk"]["id"] for result in results if "id" in result["chunk"]
            }

            if expected_ids.intersection(retrieved_ids):
                successful += 1

        return successful / len(test_cases)

    def mrr(self, test_cases):
        """
        Calculate Mean Reciprocal Rank.

        Measures the position of the first relevant result.
        """

        if not test_cases:
            return 0.0

        reciprocal_ranks = []

        for test_case in test_cases:
            query = test_case["query"]
            expected_ids = set(test_case["expected_chunk_ids"])

            results = self.retriever.search(query, top_k=len(self.retriever.chunks))

            reciprocal_rank = 0.0

            for rank, result in enumerate(results, start=1):
                chunk = result["chunk"]

                if chunk.get("id") in expected_ids:
                    reciprocal_rank = 1.0 / rank
                    break

            reciprocal_ranks.append(reciprocal_rank)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)
