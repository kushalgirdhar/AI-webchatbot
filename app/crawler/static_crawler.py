import requests
import re
from bs4 import BeautifulSoup


def fetch_page(url):
    response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})

    response.raise_for_status()

    return response.text


def parse_html(html):
    return BeautifulSoup(html, "html.parser")


def find_main_content(soup):
    return soup.find("div", class_="mw-parser-output")


def clean_text(element):
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


def extract_basic_content(soup):
    main_content = find_main_content(soup)

    if not main_content:
        return {
            "title": "",
            "heading": "",
            "headings": [],
            "paragraphs": [],
            "lists": [],
        }

    # Remove references
    for element in main_content.select(".reference, .mw-references-wrap"):
        element.decompose()

    # Remove tables
    for element in main_content.find_all("table"):
        element.decompose()

    # Page title
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Main H1 heading
    h1 = soup.find("h1")
    heading = h1.get_text(" ", strip=True) if h1 else ""

    # Extract headings
    headings = [
        clean_text(h)
        for h in main_content.find_all(["h2", "h3", "h4"])
        if h.get_text(strip=True)
    ]

    # Extract paragraphs
    paragraphs = [
        clean_text(p) for p in main_content.find_all("p") if p.get_text(strip=True)
    ]

    # Extract lists
    lists = []

    for list_tag in main_content.find_all(["ul", "ol"]):
        items = [
            clean_text(li)
            for li in list_tag.find_all("li", recursive=False)
            if li.get_text(strip=True)
        ]

        if items:
            lists.append(
                {
                    "type": list_tag.name,
                    "items": items,
                }
            )

    return {
        "title": title,
        "heading": heading,
        "headings": headings,
        "paragraphs": paragraphs,
        "lists": lists,
    }


def extract_ordered_content(soup):
    main_content = find_main_content(soup)

    if not main_content:
        return []

    # Remove references
    for element in main_content.select(".reference, .mw-references-wrap"):
        element.decompose()

    # Remove tables
    for element in main_content.find_all("table"):
        element.decompose()

    content = []

    # Keep the original order of headings, paragraphs and lists
    for element in main_content.find_all(["h2", "h3", "h4", "p", "ul", "ol"]):
        # Headings
        if element.name in ["h2", "h3", "h4"]:
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
                if li.get_text(strip=True)
            ]

            if items:
                content.append(
                    {
                        "type": "list",
                        "list_type": element.name,
                        "items": items,
                    }
                )

    return content
