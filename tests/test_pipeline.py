import unittest
from unittest.mock import MagicMock

from app.chunking.chunker import create_chunks_from_section, parse_markdown_sections
from app.cleaner.text_cleaner import clean_markdown
from app.markdown.converter import convert_to_markdown
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.e5_hybrid_retriever import E5HybridRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import DeterministicReranker
from app.retrieval.term_coverage import calculate_term_coverage
from app.retrieval.text_processor import STOP_WORDS, tokenize


class TestTextProcessor(unittest.TestCase):
    def test_tokenize_basic(self):
        tokens = tokenize("What is an ocean?")
        self.assertEqual(tokens, ["ocean"])

    def test_tokenize_hyphenated(self):
        tokens = tokenize("non-domesticated plants and micro-organisms")
        self.assertIn("non-domesticate", tokens)
        self.assertIn("micro-organism", tokens)

    def test_stop_words_filtered(self):
        tokens = tokenize("What does wildlife include?")
        self.assertEqual(tokens, ["wildlife", "include"])

    def test_pronouns_and_auxiliaries_filtered(self):
        tokens = tokenize("How would you describe an ocean?")
        self.assertEqual(tokens, ["describe", "ocean"])


class TestTermCoverage(unittest.TestCase):
    def test_full_coverage(self):
        cov = calculate_term_coverage("ocean water", "An ocean is made of water")
        self.assertEqual(cov, 1.0)

    def test_partial_coverage(self):
        cov = calculate_term_coverage("ocean mountain", "An ocean is deep")
        self.assertEqual(cov, 0.5)

    def test_empty_query(self):
        cov = calculate_term_coverage("", "Some text here")
        self.assertEqual(cov, 0.0)


class TestTextCleaner(unittest.TestCase):
    def test_clean_spaces_and_punctuation(self):
        raw = "Hello  world .This is a test ( with spaces ) ."
        cleaned = clean_markdown(raw)
        self.assertNotIn(" .", cleaned)
        self.assertNotIn("( ", cleaned)
        self.assertNotIn(" )", cleaned)

    def test_line_endings(self):
        raw = "Line 1\r\n\r\n\r\nLine 2"
        cleaned = clean_markdown(raw)
        self.assertEqual(cleaned, "Line 1\n\nLine 2")


class TestMarkdownConverter(unittest.TestCase):
    def test_conversion(self):
        content = [
            {"type": "heading", "level": 1, "text": "Main Heading"},
            {"type": "paragraph", "text": "This is a paragraph."},
            {"type": "list", "list_type": "ul", "items": ["Item 1", "Item 2"]},
        ]
        md = convert_to_markdown(content, title="Test Page")
        self.assertIn("# Test Page", md)
        self.assertIn("# Main Heading", md)
        self.assertIn("This is a paragraph.", md)
        self.assertIn("- Item 1", md)


class TestChunker(unittest.TestCase):
    def test_parse_markdown_sections(self):
        md = "# Doc Title\n\nIntroductory text.\n\n## Section 1\n\nSection 1 text.\n\n### Sub 1.1\n\nSub text."
        sections = parse_markdown_sections(md)
        self.assertTrue(len(sections) >= 2)
        self.assertEqual(sections[0]["section"], "Introduction")
        self.assertEqual(sections[1]["section"], "Section 1")

    def test_create_chunks(self):
        section = {
            "title": "Test Title",
            "section": "Section 1",
            "blocks": [
                {
                    "type": "heading",
                    "level": 2,
                    "text": "Section 1",
                    "heading_path": ["Test Title", "Section 1"],
                },
                {
                    "type": "content",
                    "text": "This is content for section 1.",
                    "heading_path": ["Test Title", "Section 1"],
                },
            ],
        }
        chunks = create_chunks_from_section(section, chunk_size=200, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Section 1", chunks[0]["text"])
        self.assertEqual(chunks[0]["metadata"]["section"], "Section 1")


class TestRetrievalAndReranker(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": 0,
                "text": "Solar Energy\n\nA solar panel is a device that converts sunlight into electricity.",
                "metadata": {
                    "title": "Solar Energy",
                    "section": "Solar Panels",
                    "heading_path": ["Solar Energy", "Solar Panels"],
                },
            },
            {
                "id": 1,
                "text": "Solar Energy > History\n\nSolar panels solar panels solar panels were developed over many decades by multiple inventors.",
                "metadata": {
                    "title": "Solar Energy",
                    "section": "History",
                    "heading_path": ["Solar Energy", "History"],
                },
            },
            {
                "id": 2,
                "text": "Solar Energy > Environmental Impact\n\nHuman activity and manufacturing of panels has significant impacts on local ecosystems.",
                "metadata": {
                    "title": "Solar Energy",
                    "section": "Environmental Impact",
                    "heading_path": ["Solar Energy", "Environmental Impact"],
                },
            },
        ]
        self.retriever = BM25Retriever(self.chunks)
        self.reranker = DeterministicReranker()

    def test_reranker_prioritizes_definition_over_repetition(self):
        query = "What is a solar panel?"
        candidates = self.retriever.search(query, top_k=3)
        reranked = self.reranker.rerank(query, candidates, top_k=2)

        self.assertEqual(reranked[0]["chunk"]["id"], 0)
        self.assertTrue(reranked[0]["definition_score"] > 0)

    def test_query_intent_detection(self):
        self.assertEqual(
            DeterministicReranker.detect_query_intent("How is climate different from weather?"),
            "COMPARISON",
        )
        self.assertEqual(
            DeterministicReranker.detect_query_intent("What effects do human activities have on water?"),
            "CAUSE_EFFECT",
        )
        self.assertEqual(
            DeterministicReranker.detect_query_intent("Where does the majority of water occur?"),
            "LOCATION",
        )
        self.assertEqual(
            DeterministicReranker.detect_query_intent("How is the natural environment defined?"),
            "DEFINITION",
        )

    def test_cause_effect_intent_selects_impact_chunk(self):
        query = "How do human activities affect the environment?"
        candidates = self.retriever.search(query, top_k=3)
        reranked = self.reranker.rerank(query, candidates, top_k=2)
        self.assertEqual(reranked[0]["chunk"]["id"], 2)

    def test_out_of_scope_query_rejection(self):
        query = "What is quantum computing?"
        candidates = self.retriever.search(query, top_k=3)
        reranked = self.reranker.rerank(query, candidates, top_k=2)
        self.assertEqual(len(reranked), 0)

    def test_partial_out_of_scope_query_rejection(self):
        query = "What is artificial intelligence?"
        candidates = self.retriever.search(query, top_k=3)
        reranked = self.reranker.rerank(query, candidates, top_k=2)
        self.assertEqual(len(reranked), 0)

    def test_empty_and_invalid_queries(self):
        for invalid_q in ["", "   ", "?", ".", "the", "what is", "and or the"]:
            candidates = self.retriever.search(invalid_q, top_k=3)
            reranked = self.reranker.rerank(invalid_q, candidates, top_k=2)
            self.assertEqual(len(candidates), 0)
            self.assertEqual(len(reranked), 0)


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": 10,
                "text": "Geological Activity\n\nThe lithosphere is the outermost solid crust of the planet.",
                "metadata": {
                    "title": "Natural Environment",
                    "section": "Geological Activity",
                    "heading_path": ["Natural Environment", "Geological Activity"],
                },
            },
            {
                "id": 20,
                "text": "Cycles\n\nThe oxygen cycle occurs through the lithosphere and biosphere.",
                "metadata": {
                    "title": "Natural Environment",
                    "section": "Cycles",
                    "heading_path": ["Natural Environment", "Cycles"],
                },
            },
        ]

    def test_rrf_scoring_math(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        bm25_res = [
            {"chunk": self.chunks[1], "score": 4.5},
            {"chunk": self.chunks[0], "score": 3.0},
        ]

        qdrant_p1 = MagicMock(id=10, score=0.85, payload={"text": self.chunks[0]["text"], "metadata": self.chunks[0]["metadata"]})
        qdrant_p2 = MagicMock(id=20, score=0.40, payload={"text": self.chunks[1]["text"], "metadata": self.chunks[1]["metadata"]})
        qdrant_res = [qdrant_p1, qdrant_p2]

        fused = retriever.fuse_rrf(bm25_res, qdrant_res, rrf_k=60)

        expected_10 = (1.0 / 62) + (1.0 / 61)
        expected_20 = (1.0 / 61) + (1.0 / 62)

        self.assertAlmostEqual(fused[0]["rrf_score"], expected_10, places=6)
        self.assertAlmostEqual(fused[1]["rrf_score"], expected_20, places=6)

    def test_rrf_prioritizes_dual_matches(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.rrf_k = 60

        bm25_res = [
            {"chunk": self.chunks[1], "score": 4.5},
            {"chunk": self.chunks[0], "score": 3.0},
        ]
        qdrant_p1 = MagicMock(id=10, score=0.90, payload={"text": self.chunks[0]["text"], "metadata": self.chunks[0]["metadata"]})
        qdrant_res = [qdrant_p1]

        fused = retriever.fuse_rrf(bm25_res, qdrant_res, rrf_k=60)

        self.assertEqual(fused[0]["chunk"]["id"], 10)
        self.assertEqual(fused[1]["chunk"]["id"], 20)
        self.assertTrue(fused[0]["rrf_score"] > fused[1]["rrf_score"])


class TestE5HybridRetriever(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": 10,
                "text": "Geological Activity\n\nThe lithosphere is the outermost solid crust of the planet.",
                "metadata": {
                    "title": "Natural Environment",
                    "section": "Geological Activity",
                    "heading_path": ["Natural Environment", "Geological Activity"],
                },
            },
            {
                "id": 20,
                "text": "Cycles\n\nThe oxygen cycle occurs through the lithosphere and biosphere.",
                "metadata": {
                    "title": "Natural Environment",
                    "section": "Cycles",
                    "heading_path": ["Natural Environment", "Cycles"],
                },
            },
        ]

    def test_e5_rrf_scoring_math(self):
        retriever = E5HybridRetriever.__new__(E5HybridRetriever)
        retriever.rrf_k = 60

        bm25_res = [
            {"chunk": self.chunks[1], "score": 4.5},
            {"chunk": self.chunks[0], "score": 3.0},
        ]

        e5_p1 = MagicMock(id=10, score=0.88, payload={"text": self.chunks[0]["text"], "metadata": self.chunks[0]["metadata"]})
        e5_p2 = MagicMock(id=20, score=0.45, payload={"text": self.chunks[1]["text"], "metadata": self.chunks[1]["metadata"]})
        e5_res = [e5_p1, e5_p2]

        fused = retriever.fuse_rrf(bm25_res, e5_res, rrf_k=60)

        expected_10 = (1.0 / 62) + (1.0 / 61)
        expected_20 = (1.0 / 61) + (1.0 / 62)

        self.assertAlmostEqual(fused[0]["rrf_score"], expected_10, places=6)
        self.assertAlmostEqual(fused[1]["rrf_score"], expected_20, places=6)


class TestPipelineProcessUrl(unittest.TestCase):
    def test_pipeline_selects_e5_by_default(self):
        from unittest.mock import patch
        from app.pipeline import process_url

        mock_content = [{"type": "paragraph", "text": "Test content about ecosystems."}]
        with patch("app.pipeline.fetch_page", return_value="<html><body><p>Test</p></body></html>"), \
             patch("app.pipeline.extract_ordered_content", return_value=mock_content), \
             patch("app.pipeline.E5EmbeddingModel") as mock_e5_embedder, \
             patch("app.pipeline.E5QdrantStore") as mock_e5_store, \
             patch("app.pipeline.E5Retriever") as mock_e5_retriever, \
             patch("app.pipeline.E5HybridRetriever") as mock_e5_hybrid:
            
            import numpy as np
            mock_e5_instance = MagicMock()
            mock_e5_instance.encode_documents.return_value = np.zeros((1, 768))
            mock_e5_embedder.return_value = mock_e5_instance
            
            result = process_url("https://example.com/test", verbose=False)
            self.assertEqual(result["embedding_model"], "multilingual-e5-base")
            self.assertTrue(mock_e5_embedder.called)
            self.assertTrue(mock_e5_store.called)

    def test_pipeline_supports_bge_selection(self):
        from unittest.mock import patch
        from app.pipeline import process_url

        mock_content = [{"type": "paragraph", "text": "Test content about ecosystems."}]
        with patch("app.pipeline.fetch_page", return_value="<html><body><p>Test</p></body></html>"), \
             patch("app.pipeline.extract_ordered_content", return_value=mock_content), \
             patch("app.pipeline.EmbeddingModel") as mock_bge_embedder, \
             patch("app.pipeline.QdrantStore") as mock_bge_store, \
             patch("app.pipeline.QdrantRetriever") as mock_bge_retriever, \
             patch("app.pipeline.HybridRetriever") as mock_bge_hybrid:
            
            import numpy as np
            mock_bge_instance = MagicMock()
            mock_bge_instance.encode.return_value = np.zeros((1, 1024))
            mock_bge_embedder.return_value = mock_bge_instance
            
            result = process_url("https://example.com/test", embedding_model="bge", verbose=False)
            self.assertEqual(result["embedding_model"], "BAAI/bge-m3")
            self.assertTrue(mock_bge_embedder.called)
            self.assertTrue(mock_bge_store.called)


if __name__ == "__main__":
    unittest.main()
