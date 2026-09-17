import re


def clean_markdown(text):
    """
    Clean extracted Markdown/text without changing its meaning.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove Wikipedia citation/template artifacts
    text = re.sub(
        r"\{\{\s*cite book\s*\}\}:\s*CS1 maint:\s*location\s*\(\s*link\s*\)",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove empty lines with trailing spaces
    text = re.sub(r"[ \t]+\n", "\n", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Remove spaces before closing parenthesis
    text = re.sub(r"\s+\)", ")", text)

    # Remove spaces after opening parenthesis
    text = re.sub(r"\(\s+", "(", text)

    # Remove spaces after hyphen inside words
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)

    # Ensure space after punctuation when followed by a letter
    text = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r",(?=[A-Za-z])", ", ", text)

    return text.strip()