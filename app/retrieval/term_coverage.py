from app.retrieval.text_processor import tokenize


def calculate_term_coverage(query, text):
    """
    Calculate how many unique query terms appear in the text.

    Returns a value between 0.0 and 1.0.
    """

    query_tokens = set(tokenize(query))
    text_tokens = set(tokenize(text))

    if not query_tokens:
        return 0.0

    matched_terms = query_tokens.intersection(
        text_tokens
    )

    return len(matched_terms) / len(query_tokens)