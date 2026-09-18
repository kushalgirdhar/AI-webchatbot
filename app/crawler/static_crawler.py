import re
import requests
from bs4 import BeautifulSoup


def fetch_page(url):
    response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    return response.text


def parse_html(html):
    return BeautifulSoup(html, "html.parser")


def find_main_content(soup):
    candidates = []

    # 1. Wikipedia main content containers
    for div in soup.find_all("div", class_="mw-parser-output"):
        txt_len = len(div.get_text(strip=True))
        if div.find_all(["p", "h1", "h2", "h3", "ul", "ol"]) or txt_len > 100:
            candidates.append((txt_len, div))

    # 2. Semantic article or main container
    for tag_name in ["article", "main"]:
        for el in soup.find_all(tag_name):
            txt_len = len(el.get_text(strip=True))
            if txt_len > 100:
                candidates.append((txt_len, el))

    # 3. Common content container IDs and classes
    for div in soup.find_all(
        "div",
        id=re.compile(
            r"^(bodyContent|content|main-content|main|article|post|body-content)$",
            re.IGNORECASE,
        ),
    ):
        txt_len = len(div.get_text(strip=True))
        if txt_len > 100:
            candidates.append((txt_len, div))

    for div in soup.find_all(
        "div",
        class_=re.compile(
            r"\b(content|main-content|article-content|post-content|entry-content|vector-body)\b",
            re.IGNORECASE,
        ),
    ):
        txt_len = len(div.get_text(strip=True))
        if txt_len > 100:
            candidates.append((txt_len, div))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # 4. Fallback to body or entire soup
    return soup.body if soup.body else soup


def clean_text(element):
    if not element:
        return ""

    # Get text from individual HTML text nodes
    parts = []
    for text_node in element.stripped_strings:
        parts.append(text_node)

    # Join separate text nodes with spaces
    text = " ".join(parts)

    # Normalize whitespace
    text = " ".join(text.split())

    # Remove spaces before punctuation
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    text = text.replace(" ;", ";")
    text = text.replace(" :", ":")
    text = text.replace(" !", "!")
    text = text.replace(" ?", "?")
    text = text.replace(" )", ")")

    # Normalize spaces after opening parentheses
    text = text.replace("( ", "(")

    # Remove spaces after hyphens inside words
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "-", text)

    # Ensure a space after sentence punctuation when followed by a word
    text = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)

    # Ensure a space after comma when followed by a word
    text = re.sub(r",(?=[A-Za-z])", ", ", text)

    return text


def extract_ordered_content(soup):
    main_content = find_main_content(soup)
    if not main_content:
        return []

    # Clean out unwanted noise tags
    for tag in main_content.find_all(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript",
            "iframe",
            "svg",
        ]
    ):
        tag.decompose()

    # Remove citations / references
    for element in main_content.select(".reference, .mw-references-wrap"):
        element.decompose()

    # Remove tables
    for element in main_content.find_all("table"):
        element.decompose()

    content = []

    # Keep the original order of headings, paragraphs and lists
    for element in main_content.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"]
    ):
        # Headings
        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            text = clean_text(element)
            if text:
                content.append(
                    {
                        "type": "heading",
                        "level": int(element.name[1]),
                        "text": text,
                    }
                )

        # Paragraphs
        elif element.name == "p":
            text = clean_text(element)
            if text:
                content.append(
                    {
                        "type": "paragraph",
                        "text": text,
                    }
                )

        # Lists
        elif element.name in ["ul", "ol"]:
            items = [
                clean_text(li)
                for li in element.find_all("li", recursive=False)
                if clean_text(li)
            ]
            if items:
                content.append(
                    {
                        "type": "list",
                        "list_type": element.name,
                        "items": items,
                    }
                )

    # Fallback if no structured tags found
    if not content:
        text = clean_text(main_content)
        if text:
            content.append({"type": "paragraph", "text": text})

    return content
