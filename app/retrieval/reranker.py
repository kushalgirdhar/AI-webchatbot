import re

from app.retrieval.text_processor import STOP_WORDS, tokenize


class DeterministicReranker:
    """
    General-purpose deterministic reranker combining:
    1. Baseline BM25 lexical relevance
    2. Term coverage & exact phrase co-occurrence
    3. Structural & section heading alignment
    4. Subject-focused definitional and equivalence patterns
    5. Lead sentence prominence
    """

    TERM_COVERAGE_WEIGHT = 1.0
    EXACT_PHRASE_WEIGHT = 0.8
    DEFINITION_WEIGHT = 1.5
    HEADING_WEIGHT = 0.8
    LEAD_WEIGHT = 0.5

    DEF_COPULAS = (
        r"(?<!there\s)(?<!where\s)(?<!here\s)\b"
        r"(?:is|are|refers to|is defined as|is described as|defined as|described as|"
        r"consists of|includes|include|comprises|comprise|encompasses|encompass|"
        r"serves as|acts as|can be defined as|constitutes|constitute|means|mean|"
        r"looks at|examines|deals with|covers|describes)\b"
    )

    def rerank(self, query, results, top_k=5):
        """
        Rerank BM25 candidate results deterministically.
        """
        reranked_results = []
        query_content_tokens = self._get_content_tokens(query)

        for result in results:
            chunk = result["chunk"]
            bm25_score = result.get("score", 0.0)
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            # 1. Term coverage
            coverage = self._calculate_term_coverage(query_content_tokens, text)

            # 2. Exact phrase co-occurrence
            phrase_score = self._calculate_phrase_score(query_content_tokens, text)

            # 3. Heading & section alignment
            heading_score = self._calculate_heading_score(
                query_content_tokens, metadata
            )

            # 4. Clean sentences for text analysis
            sentences = self._extract_clean_sentences(text)

            # 5. Definitional sentence match
            def_score = self._calculate_definition_score(
                query_content_tokens, sentences
            )

            # 6. Lead sentence prominence
            lead_score = self._calculate_lead_score(query_content_tokens, sentences)

            final_score = (
                bm25_score
                + (coverage * self.TERM_COVERAGE_WEIGHT)
                + (phrase_score * self.EXACT_PHRASE_WEIGHT)
                + (def_score * self.DEFINITION_WEIGHT)
                + (heading_score * self.HEADING_WEIGHT)
                + (lead_score * self.LEAD_WEIGHT)
            )

            reranked_results.append(
                {
                    "chunk": chunk,
                    "score": final_score,
                    "bm25_score": bm25_score,
                    "term_coverage": coverage,
                    "exact_phrase_score": phrase_score,
                    "definition_score": def_score,
                    "heading_score": heading_score,
                    "lead_score": lead_score,
                }
            )

        reranked_results.sort(key=lambda r: r["score"], reverse=True)
        return reranked_results[:top_k]

    @staticmethod
    def _get_content_tokens(text):
        """
        Extract non-stopword normalized tokens from text.
        """
        return [token for token in tokenize(text) if token not in STOP_WORDS]

    @staticmethod
    def _extract_clean_sentences(text):
        """
        Strip leading breadcrumbs and split text into clean sentences.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned_lines = []
        for line in lines:
            if " > " in line and len(line.split(" > ")) >= 2:
                continue
            cleaned_lines.append(line)

        clean_text = " ".join(cleaned_lines)
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _calculate_term_coverage(query_tokens, text):
        if not query_tokens:
            return 0.0

        text_tokens = set(tokenize(text))
        matched = [t for t in query_tokens if t in text_tokens]
        return len(matched) / len(query_tokens)

    @staticmethod
    def _calculate_phrase_score(query_tokens, text):
        if len(query_tokens) < 2:
            return 0.0

        text_lower = text.lower()
        exact_phrase = " ".join(query_tokens)

        if exact_phrase in text_lower:
            return 1.0

        loose_pattern = r"\b" + r"\s+(?:\w+\s+){0,2}".join(
            re.escape(t) for t in query_tokens
        ) + r"\b"
        if re.search(loose_pattern, text_lower):
            return 0.6

        return 0.0

    @staticmethod
    def _calculate_heading_score(query_tokens, metadata):
        if not query_tokens or not metadata:
            return 0.0

        section = metadata.get("section", "").lower()
        heading_path = [h.lower() for h in metadata.get("heading_path", [])]

        score = 0.0
        for token in query_tokens:
            if heading_path and token in heading_path[-1]:
                score += 1.0
            elif token in section:
                score += 0.5
            elif any(token in h for h in heading_path):
                score += 0.3

        return score

    def _calculate_definition_score(self, query_tokens, sentences):
        if not query_tokens or not sentences:
            return 0.0

        max_score = 0.0

        for idx, sentence in enumerate(sentences[:4]):
            s_lower = sentence.lower()
            s_clean = re.sub(r"\([^)]*\)", "", s_lower).strip()

            for token in query_tokens:
                # Subject definition: target at start of clause followed by copula/def verb
                p_subject = (
                    rf"^(?:an?|the)?\s*(?:[a-z0-9\s,\'\"\-]{{0,30}}\b{re.escape(token)}\b[a-z0-9\s,\'\"\-]{{0,30}})\s+{self.DEF_COPULAS}"
                )
                if re.search(p_subject, s_clean) or re.search(p_subject, s_lower):
                    b = 2.5 if idx == 0 else (1.5 if idx == 1 else 0.8)
                    max_score = max(max_score, b)
                    continue

                # Clause-level subject definition after punctuation/conjunction
                p_clause = (
                    rf"[,;:\-—]\s*(?:an?|the)?\s*(?:[a-z0-9\s,\'\"\-]{{0,25}}\b{re.escape(token)}\b[a-z0-9\s,\'\"\-]{{0,25}})\s+{self.DEF_COPULAS}"
                )
                if re.search(p_clause, s_clean) or re.search(p_clause, s_lower):
                    b = 1.8 if idx == 0 else (1.2 if idx == 1 else 0.6)
                    max_score = max(max_score, b)
                    continue

                # Passive definition: ...is called / known as / defined as <target>
                p_passive = (
                    rf"\b(?:is|are|called|known as|defined as|described as)\s+(?:an?\s+|the\s+)?(?:[a-z0-9\s,\'\"\-]{{0,25}}\s+)?\b{re.escape(token)}\b"
                )
                if re.search(p_passive, s_clean) or re.search(p_passive, s_lower):
                    b = 1.5 if idx == 0 else (1.0 if idx == 1 else 0.5)
                    max_score = max(max_score, b)

        return max_score

    @staticmethod
    def _calculate_lead_score(query_tokens, sentences):
        if not query_tokens or not sentences:
            return 0.0

        lead_sentence = sentences[0].lower()
        matched = sum(1 for token in query_tokens if token in lead_sentence)
        return matched / len(query_tokens)


# Backwards compatibility alias
DefinitionReranker = DeterministicReranker