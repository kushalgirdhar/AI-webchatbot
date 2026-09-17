import json

from app.chunking.chunker import (
    parse_markdown_sections,
    create_chunks_from_section,
)


INPUT_FILE = "data/markdown/natural_environment.md"
OUTPUT_FILE = "data/chunks/natural_environment.json"


def main():
    # Read Markdown
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    # Parse Markdown into sections
    sections = parse_markdown_sections(markdown_text)

    # Create chunks
    all_chunks = []

    for section in sections:
        chunks = create_chunks_from_section(
            section,
            chunk_size=800,
            overlap=100
        )

        all_chunks.extend(chunks)

    # Save chunks
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Chunks saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()