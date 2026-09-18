import re

from app.retrieval.text_processor import STOP_WORDS, normalize_token, tokenize

META_RAW = [
    "define",
    "defined",
    "definition",
    "definitions",
    "describe",
    "described",
    "description",
    "characterize",
    "characterized",
    "characterizes",
    "characteristic",
    "characteristics",
    "mean",
    "means",
    "meaning",
    "meanings",
    "refer",
    "refers",
    "referred",
    "distinguish",
    "distinguishes",
    "distinguished",
    "difference",
    "differences",
    "different",
    "consider",
    "considered",
    "considers",
    "term",
    "terms",
    "concept",
    "concepts",
    "occur",
    "occurs",
    "occurred",
    "found",
    "find",
    "make",
    "made",
    "many",
    "much",
]
META_INTENT_WORDS = {normalize_token(w) for w in META_RAW}.union(set(META_RAW))

ACTION_RAW = [
    "effect",
    "effects",
    "affect",
    "affects",
    "impact",
    "impacts",
    "influence",
    "influences",
    "cause",
    "causes",
    "human",
    "activities",
    "activity",
    "threat",
    "threats",
    "threaten",
    "damage",
]
ACTION_CAUSE_WORDS = {normalize_token(w) for w in ACTION_RAW}.union(set(ACTION_RAW))


class DeterministicReranker:
    """
    General-purpose, model-free deterministic reranker combining:
    1. Baseline BM25 lexical score
    2. Query intent detection (Definition, Cause/Effect, Comparison, Location, Quantity, General)
    3. Subject entity coverage & full phrase matching
    4. Structural & section heading alignment
    5. Subject-focused definitional and equivalence patterns with positional decay
    6. Intent-specific semantic action & comparison matching
    7. Lead sentence prominence
    8. Relevance thresholding & out-of-scope query rejection
    """

    TERM_COVERAGE_WEIGHT = 1.0
    SUBJECT_COVERAGE_WEIGHT = 1.0
    FULL_SUBJECT_WEIGHT = 3.0
    EXACT_PHRASE_WEIGHT = 1.0
    DEFINITION_WEIGHT = 2.0
    HEADING_WEIGHT = 1.0
    LEAD_WEIGHT = 0.5
    ACTION_WEIGHT = 1.2

    DEF_COPULAS = (
        r"(?<!there\s)(?<!where\s)(?<!here\s)\b"
        r"(?:is|are|refers to|is defined as|is described as|defined as|described as|"
        r"consists of|includes|include|comprises|comprise|encompasses|encompass|"
        r"divided into|divided|can be divided into|split into|classified into|"
        r"serves as|acts as|can be defined as|constitutes|constitute|means|mean|"
        r"looks at|examines|deals with|covers|describes)\b"
    )

    def rerank(self, query, results, top_k=5):
        """
        Rerank BM25 candidate results deterministically.
        Returns an empty list for empty, invalid, or out-of-scope queries.
        """
        content_tokens = tokenize(query)
        if not content_tokens:
            return []

        if not results:
            return []

        # If highest candidate BM25 score is 0.0, indexed collection has no matching terms
        if max(result.get("score", 0.0) for result in results) <= 0.0:
            return []

        subject_tokens = self.get_subject_tokens(query)
        intent = self.detect_query_intent(query)
        subj_phrase = " ".join(subject_tokens)

        reranked_results = []

        for result in results:
            chunk = result["chunk"]
            bm25_score = result.get("score", 0.0)
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            section = metadata.get("section", "").lower()
            heading_path = [h.lower() for h in metadata.get("heading_path", [])]

            text_tokens = set(tokenize(text))
            meta_tokens = set(tokenize(" ".join([section] + heading_path)))
            all_doc_tokens = text_tokens.union(meta_tokens)

            # 1. Content and subject term coverage
            matched_content = [t for t in content_tokens if t in all_doc_tokens]
            content_cov = len(matched_content) / len(content_tokens)

            matched_subject = [t for t in subject_tokens if t in all_doc_tokens]
            subject_cov = (
                len(matched_subject) / len(subject_tokens) if subject_tokens else 1.0
            )

            # Full subject coverage bonus
            full_subject_bonus = (
                2.0 if subject_cov == 1.0 and len(subject_tokens) >= 2 else 0.0
            )

            # 2. Phrase match
            phrase_score = self._calculate_phrase_score(subject_tokens, text)

            # 3. Heading alignment
            heading_score = self._calculate_heading_score(
                subject_tokens, section, heading_path
            )

            # 4. Clean sentences for text analysis
            sentences = self._extract_clean_sentences(chunk)

            # 5. Definition score (applied primarily when query intent is DEFINITION, COMPARISON, or GENERAL)
            def_score = self._calculate_definition_score(
                subject_tokens, subj_phrase, sentences, intent
            )

            # 6. Intent-specific Action / Comparison / Location score
            action_score = self._calculate_intent_score(
                intent, text, text_tokens, section, heading_path
            )

            # 7. Lead sentence prominence
            lead_score = self._calculate_lead_score(subject_tokens, sentences)

            final_score = (
                bm25_score
                + (content_cov * self.TERM_COVERAGE_WEIGHT)
                + (subject_cov * self.SUBJECT_COVERAGE_WEIGHT)
                + (full_subject_bonus * self.FULL_SUBJECT_WEIGHT)
                + (phrase_score * self.EXACT_PHRASE_WEIGHT)
                + (def_score * self.DEFINITION_WEIGHT)
                + (heading_score * self.HEADING_WEIGHT)
                + (lead_score * self.LEAD_WEIGHT)
                + (action_score * self.ACTION_WEIGHT)
            )

            reranked_results.append(
                {
                    "chunk": chunk,
                    "score": final_score,
                    "bm25_score": bm25_score,
                    "content_coverage": content_cov,
                    "subject_coverage": subject_cov,
                    "phrase_score": phrase_score,
                    "definition_score": def_score,
                    "heading_score": heading_score,
                    "action_score": action_score,
                    "lead_score": lead_score,
                }
            )

        reranked_results.sort(key=lambda r: r["score"], reverse=True)

        # Relevance Threshold & Out-of-Scope Filter
        if reranked_results:
            top_res = reranked_results[0]
            if top_res["bm25_score"] <= 0.0:
                return []
            # Specific 2-word entities (like "artificial intelligence") must have full coverage or phrase match
            if (
                len(subject_tokens) == 2
                and top_res["subject_coverage"] < 1.0
                and top_res["phrase_score"] == 0.0
                and top_res["bm25_score"] < 4.0
            ):
                return []
            # Multi-word queries with extremely low coverage across the entire document
            if len(subject_tokens) > 2 and top_res["subject_coverage"] < 0.35:
                return []

        return reranked_results[:top_k]

    @staticmethod
    def detect_query_intent(query):
        """
        Detect question intent from grammatical and lexical indicators.
        """
        q_lower = query.lower()
        if any(
            w in q_lower
            for w in ["different", "difference", "distinguish", "versus", "vs"]
        ):
            return "COMPARISON"
        if any(
            w in q_lower
            for w in [
                "affect",
                "affects",
                "effect",
                "effects",
                "impact",
                "impacts",
                "influence",
                "influences",
                "cause",
                "causes",
                "human activities",
                "human activity",
            ]
        ):
            return "CAUSE_EFFECT"
        if any(w in q_lower for w in ["where", "where does", "where is", "location"]):
            return "LOCATION"
        if any(
            w in q_lower
            for w in ["how many", "how much", "number of", "percentage", "amount"]
        ):
            return "QUANTITY"
        if any(
            w in q_lower
            for w in [
                "what is",
                "what are",
                "what does",
                "how is",
                "how would",
                "define",
                "defined",
                "definition",
                "describe",
                "described",
                "characterize",
                "characterizes",
                "characterized",
                "mean",
                "refer",
            ]
        ):
            return "DEFINITION"
        return "GENERAL"

    @staticmethod
    def get_subject_tokens(query):
        """
        Extract core subject tokens by excluding question meta-intent words.
        """
        tokens = tokenize(query)
        subj = [t for t in tokens if t not in META_INTENT_WORDS]
        return subj if subj else tokens

    @staticmethod
    def _extract_clean_sentences(chunk):
        """
        Strip breadcrumbs, markdown headers, and standalone title lines before sentence splitting.
        """
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        title = metadata.get("title", "").strip()
        section = metadata.get("section", "").strip()

        lines = text.split("\n")
        body_lines = []
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if " > " in l or l.startswith("#"):
                continue
            if l == title or (title and l.startswith(title)):
                continue
            if l == section:
                continue
            body_lines.append(l)

        clean_text = " ".join(body_lines)
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _calculate_phrase_score(subject_tokens, text):
        if len(subject_tokens) < 2:
            return 0.0

        text_tokens = tokenize(text)
        text_joined = " ".join(text_tokens)
        subj_phrase = " ".join(subject_tokens)

        if subj_phrase in text_joined:
            return 1.0

        loose_pattern = (
            r"\b"
            + r"\s+(?:\w+\s+){0,2}".join(re.escape(t) for t in subject_tokens)
            + r"\b"
        )
        if re.search(loose_pattern, text_joined):
            return 0.6

        return 0.0

    @staticmethod
    def _calculate_heading_score(subject_tokens, section, heading_path):
        if not subject_tokens:
            return 0.0

        score = 0.0
        for token in subject_tokens:
            if heading_path and token in heading_path[-1]:
                score += 1.0
            elif token in section:
                score += 0.5
            elif any(token in h for h in heading_path):
                score += 0.3

        return score

    def _calculate_definition_score(
        self, subject_tokens, subj_phrase, sentences, intent
    ):
        if not subject_tokens or not sentences:
            return 0.0

        if intent not in ("DEFINITION", "GENERAL", "COMPARISON"):
            return 0.0

        max_score = 0.0

        for idx, sentence in enumerate(sentences[:4]):
            s_lower = sentence.lower()
            s_clean = re.sub(r"\([^)]*\)", "", s_lower).strip()

            # 1. Full subject phrase definition match (highest confidence)
            if len(subject_tokens) >= 2:
                p_full = (
                    rf"^(?:an?|the)?\s*(?:[a-z0-9\s,\'\"\-]{{0,30}}\b{re.escape(subj_phrase)}\b[a-z0-9\s,\'\"\-]{{0,30}})\s+{self.DEF_COPULAS}"
                )
                if re.search(p_full, s_clean) or re.search(p_full, s_lower):
                    b = 3.5 if idx == 0 else (2.0 if idx == 1 else 1.0)
                    max_score = max(max_score, b)
                    continue

            # 2. Individual subject token definition matches
            for token in subject_tokens:
                p_subject_exact = (
                    rf"^(?:an?|the)?\s*\b{re.escape(token)}\b\s+{self.DEF_COPULAS}"
                )
                if re.search(p_subject_exact, s_clean) or re.search(p_subject_exact, s_lower):
                    b = 3.0 if idx == 0 else (1.8 if idx == 1 else 1.0)
                    max_score = max(max_score, b)
                    continue

                p_subject = (
                    rf"^(?:an?|the)?\s*(?:[a-z0-9\s,\'\"\-]{{0,25}}\b{re.escape(token)}\b[a-z0-9\s,\'\"\-]{{0,25}})\s+{self.DEF_COPULAS}"
                )
                if re.search(p_subject, s_clean) or re.search(p_subject, s_lower):
                    b = 2.2 if idx == 0 else (1.4 if idx == 1 else 0.7)
                    max_score = max(max_score, b)
                    continue

                p_clause = (
                    rf"[,;:\-—]\s*(?:an?|the)?\s*(?:[a-z0-9\s,\'\"\-]{{0,25}}\b{re.escape(token)}\b[a-z0-9\s,\'\"\-]{{0,25}})\s+{self.DEF_COPULAS}"
                )
                if re.search(p_clause, s_clean) or re.search(p_clause, s_lower):
                    b = 1.6 if idx == 0 else (1.0 if idx == 1 else 0.5)
                    max_score = max(max_score, b)
                    continue

                p_passive = (
                    rf"\b(?:is called|are called|is known as|are known as|is termed|are termed|is referred to as|are referred to as)\s+(?:an?\s+|the\s+)?(?:[a-z0-9\s,\'\"\-]{{0,25}}\s+)?\b{re.escape(token)}\b"
                )
                if re.search(p_passive, s_clean) or re.search(p_passive, s_lower):
                    b = 1.5 if idx == 0 else (1.0 if idx == 1 else 0.5)
                    max_score = max(max_score, b)

        return max_score

    @staticmethod
    def _calculate_intent_score(
        intent, text, text_tokens, section, heading_path
    ):
        if intent == "CAUSE_EFFECT":
            action_matches = sum(
                1
                for w in ACTION_CAUSE_WORDS
                if w in text_tokens or any(w in h for h in heading_path)
            )
            score = 0.0
            if any(
                w in section or any(w in h for h in heading_path)
                for w in ["impact", "human", "effect", "challenge"]
            ):
                score += 2.0
            score += min(action_matches * 0.4, 1.5)
            return score

        elif intent == "COMPARISON":
            text_lower = text.lower()
            if any(
                w in text_lower
                for w in [
                    "on the other hand",
                    "whereas",
                    "while",
                    "unlike",
                    "contrast",
                    "different",
                    "difference",
                ]
            ):
                return 2.0

        elif intent == "LOCATION":
            text_lower = text.lower()
            if any(
                w in text_lower
                for w in [
                    "found in",
                    "located in",
                    "covers",
                    "majority of",
                    "most water",
                ]
            ):
                return 1.5

        return 0.0

    @staticmethod
    def _calculate_lead_score(subject_tokens, sentences):
        if not subject_tokens or not sentences:
            return 0.0

        lead_sentence = sentences[0].lower()
        matched = sum(1 for token in subject_tokens if token in lead_sentence)
        return matched / len(subject_tokens)


# Backwards compatibility alias
DefinitionReranker = DeterministicReranker