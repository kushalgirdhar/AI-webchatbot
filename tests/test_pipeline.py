import unittest

from app.chunking.chunker import create_chunks_from_section, parse_markdown_sections
from app.cleaner.text_cleaner import clean_markdown
from app.markdown.converter import convert_to_markdown
from app.retrieval.bm25 import BM25Retriever
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
            "CAUSE_EFFECT"
        )
        self.assertEqual(
            DeterministicReranker.detect_query_intent("Where does the majority of water occur?"),
            "LOCATION"
        )
        self.assertEqual(
            DeterministicReranker.detect_query_intent("How is the natural environment defined?"),
            "DEFINITION"
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


if __name__ == "__main__":
    unittest.main()
