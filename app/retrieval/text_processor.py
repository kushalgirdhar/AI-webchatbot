import re

STOP_WORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "s",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def normalize_token(token):
    """
    Deterministic rule-based English suffix normalization (singular/plural and verb inflections).
    """
    if len(token) <= 3:
        return token

    # Handle hyphenated sub-tokens individually
    if "-" in token:
        parts = token.split("-")
        return "-".join(normalize_token(p) for p in parts)

    # Plural -ies -> -y (e.g. activities -> activity)
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    # Plural -es (e.g. processes -> process, boxes -> box)
    if token.endswith("es") and len(token) > 4 and token[-3] in ("s", "x", "z", "c", "h"):
        return token[:-2]

    # Plural -s (e.g. layers -> layer, organisms -> organism)
    if (
        token.endswith("s")
        and not token.endswith("ss")
        and not token.endswith("us")
        and not token.endswith("is")
        and len(token) > 3
    ):
        return token[:-1]

    # Participle -ing (e.g. impacting -> impact, running -> run)
    if token.endswith("ing") and len(token) > 5:
        if len(token) > 6 and token[-4] == token[-5]:
            return token[:-4]
        return token[:-3]

    # Past -ated -> -ate (e.g. domesticated -> domesticate)
    if token.endswith("ated") and len(token) > 5:
        return token[:-1]

    # Past -ed (e.g. surrounded -> surround)
    if token.endswith("ed") and len(token) > 4 and not token.endswith("eed"):
        if len(token) > 5 and token[-3] == token[-4]:
            return token[:-3]
        return token[:-2]

    return token


def tokenize(text, stem=True):
    """
    Convert text into normalized tokens for BM25 retrieval and deterministic scoring.
    """
    if not text:
        return []

    text = text.lower()
    raw_tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text)

    tokens = []
    for token in raw_tokens:
        if token in STOP_WORDS or len(token) <= 1:
            continue
        if stem:
            tokens.append(normalize_token(token))
        else:
            tokens.append(token)

    return tokens