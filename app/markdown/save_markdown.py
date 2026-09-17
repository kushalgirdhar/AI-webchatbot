from app.crawler.static_crawler import (
    fetch_page,
    parse_html,
    extract_ordered_content,
)
from app.markdown.converter import convert_to_markdown
from app.cleaner.text_cleaner import clean_markdown

URL = "https://en.wikipedia.org/wiki/Natural_environment"


def main():
    # Fetch webpage
    html = fetch_page(URL)

    # Parse HTML
    soup = parse_html(html)

    # Extract ordered content
    content = extract_ordered_content(soup)

    # Get page title
    title = soup.title.get_text(strip=True) if soup.title else ""

    # Convert extracted content to Markdown
    markdown = convert_to_markdown(content, title)
    markdown = clean_markdown(markdown)
    # Save Markdown to file

    output_file = "data/markdown/natural_environment.md"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(markdown)

    print(f"Markdown saved to: {output_file}")


if __name__ == "__main__":
    main()
