def convert_to_markdown(content, title=""):
    markdown = []

    if title:
        markdown.append(f"# {title}")
        markdown.append("")

    for block in content:
        if block["type"] == "heading":
            level = block["level"]
            text = block["text"]

            markdown.append(f"{'#' * level} {text}")
            markdown.append("")

        elif block["type"] == "paragraph":
            markdown.append(block["text"])
            markdown.append("")

        elif block["type"] == "list":
            for index, item in enumerate(block["items"], start=1):
                if block["list_type"] == "ol":
                    markdown.append(f"{index}. {item}")
                else:
                    markdown.append(f"- {item}")

            markdown.append("")

    return "\n".join(markdown).strip()