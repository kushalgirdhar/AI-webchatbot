import re


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
}


def tokenize(text):
    """
    Convert text into normalized tokens for BM25 retrieval.
    """

    # Convert to lowercase
    text = text.lower()

    # Extract words and hyphenated terms
    tokens = re.findall(
        r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b",
        text
    )

    # Remove stop words
    tokens = [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]

    return tokens