import re


def parse_markdown_sections(markdown_text):
    sections = []

    title = ""
    heading_stack = []

    current_section = None
    current_blocks = []
    current_block = []

    def save_current_block():
        if not current_block:
            return

        text = "\n".join(current_block).strip()

        if text:
            current_blocks.append(
                {"type": "content", "text": text, "heading_path": heading_stack.copy()}
            )

    def save_current_section():
        save_current_block()

        if current_section is not None or current_blocks:
            sections.append(
                {
                    "section": current_section or "Introduction",
                    "title": title,
                    "blocks": current_blocks.copy(),
                }
            )

    for line in markdown_text.splitlines():

        stripped_line = line.strip()

        # -----------------------------------------
        # Detect Markdown headings H1-H6
        # -----------------------------------------
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped_line)

        if heading_match:

            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()

            # -------------------------------------
            # H1 → Document title
            # -------------------------------------
            if level == 1:

                save_current_section()

                title = heading
                heading_stack = [title]

                current_section = None
                current_blocks = []
                current_block = []

            # -------------------------------------
            # H2 → Main section
            # -------------------------------------
            elif level == 2:

                # Save content before first H2
                # as Introduction
                save_current_block()

                if current_section is None and current_blocks:

                    sections.append(
                        {
                            "section": "Introduction",
                            "title": title,
                            "blocks": current_blocks.copy(),
                        }
                    )

                else:
                    save_current_section()

                heading_stack = [title, heading] if title else [heading]

                current_section = heading

                current_blocks = [
                    {
                        "type": "heading",
                        "level": level,
                        "text": heading,
                        "heading_path": heading_stack.copy(),
                    }
                ]

                current_block = []

            # -------------------------------------
            # H3-H6 → Subsection
            # -------------------------------------
            else:

                save_current_block()

                # Keep only parents above this level
                heading_stack = heading_stack[: level - 1]

                heading_stack.append(heading)

                current_blocks.append(
                    {
                        "type": "heading",
                        "level": level,
                        "text": heading,
                        "heading_path": heading_stack.copy(),
                    }
                )

                current_block = []

            continue

        # -----------------------------------------
        # Blank line → end current content block
        # -----------------------------------------
        if not stripped_line:

            save_current_block()
            current_block = []

            continue

        # -----------------------------------------
        # Normal content
        # -----------------------------------------
        current_block.append(stripped_line)

    # ---------------------------------------------
    # Save final section
    # ---------------------------------------------
    save_current_section()

    return sections


def create_chunks_from_section(section, chunk_size=800, overlap=100):
    chunks = []

    current_text = ""
    current_heading_path = []

    def build_context(heading_path):
        if not heading_path:
            return ""

        return " > ".join(heading_path)

    def save_chunk():
        nonlocal current_text

        if not current_text.strip():
            current_text = ""
            return

        chunks.append(
            {
                "text": current_text.strip(),
                "metadata": {
                    "title": section["title"],
                    "section": section["section"],
                    "heading_path": current_heading_path.copy(),
                },
            }
        )

        current_text = ""

    for block in section["blocks"]:

        # -----------------------------------------
        # Heading
        # -----------------------------------------
        if block["type"] == "heading":

            # Finish previous heading's content
            save_chunk()

            current_heading_path = block.get("heading_path", []).copy()

            continue

        # -----------------------------------------
        # Ignore unknown block types
        # -----------------------------------------
        if block["type"] != "content":
            continue

        block_text = block["text"].strip()

        if not block_text:
            continue

        block_heading_path = block.get("heading_path", current_heading_path).copy()

        heading_context = build_context(block_heading_path)

        # -----------------------------------------
        # Large block
        # -----------------------------------------
        if len(block_text) + len(heading_context) + 2 > chunk_size:

            save_chunk()

            text_with_context = (
                f"{heading_context}\n\n{block_text}" if heading_context else block_text
            )

            large_chunks = split_large_block(text_with_context, chunk_size, overlap)

            for large_chunk in large_chunks:

                chunks.append(
                    {
                        "text": large_chunk.strip(),
                        "metadata": {
                            "title": section["title"],
                            "section": section["section"],
                            "heading_path": block_heading_path.copy(),
                        },
                    }
                )

            current_heading_path = block_heading_path

            continue

        # -----------------------------------------
        # Start new chunk
        # -----------------------------------------
        if not current_text:

            if heading_context:
                current_text = f"{heading_context}\n\n" f"{block_text}"
            else:
                current_text = block_text

            current_heading_path = block_heading_path

            continue

        # -----------------------------------------
        # Don't mix different heading contexts
        # -----------------------------------------
        if block_heading_path != current_heading_path:

            save_chunk()

            if heading_context:
                current_text = f"{heading_context}\n\n" f"{block_text}"
            else:
                current_text = block_text

            current_heading_path = block_heading_path

            continue

        # -----------------------------------------
        # Try adding another content block
        # -----------------------------------------
        candidate = current_text + "\n\n" + block_text

        if len(candidate) <= chunk_size:

            current_text = candidate

        else:

            save_chunk()

            current_text = (
                f"{heading_context}\n\n" f"{block_text}"
                if heading_context
                else block_text
            )

            current_heading_path = block_heading_path

    # ---------------------------------------------
    # Save final chunk
    # ---------------------------------------------
    save_chunk()

    return chunks


def split_large_block(text, chunk_size=800, overlap=100):
    """
    Split a large content block into chunks.

    Strategy:

    1. Sentence-based splitting
    2. Sentence overlap
    3. Word-based splitting for very long sentences

    chunk_size = hard maximum
    overlap    = target overlap
    """

    # -----------------------------------------
    # Split text into sentences
    # -----------------------------------------
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    chunks = []

    current_sentences = []
    current_length = 0

    def sentence_length(sentence):
        return len(sentence)

    def build_chunk(sentences_list):
        return " ".join(sentences_list).strip()

    # -----------------------------------------
    # Find semantic overlap
    # -----------------------------------------
    def get_overlap_sentences(sentences_list, max_overlap_length=None):
        """
        Select recent complete sentences for overlap.

        `overlap` is the desired target.

        `max_overlap_length` prevents the overlap
        from making the next chunk exceed chunk_size.
        """

        if not sentences_list:
            return []

        if max_overlap_length is None:
            max_overlap_length = overlap

        target_length = min(overlap, max_overlap_length)

        candidates = []
        total_length = 0

        # Start from the most recent sentence
        for sentence in reversed(sentences_list):

            extra_length = len(sentence) if not candidates else len(sentence) + 1

            # Don't exceed the available space
            if total_length + extra_length > target_length:
                break

            candidates.insert(0, sentence)

            total_length += extra_length

        return candidates

    # -----------------------------------------
    # Split very long sentence by words
    # -----------------------------------------
    def split_long_sentence(sentence):
        """
        Split a sentence longer than chunk_size
        using word boundaries.
        """

        words = sentence.split()

        pieces = []
        current_words = []
        current_length = 0

        for word in words:

            # First word
            if not current_words:

                current_words = [word]
                current_length = len(word)

                continue

            new_length = current_length + 1 + len(word)

            # Word fits
            if new_length <= chunk_size:

                current_words.append(word)
                current_length = new_length

            # Word doesn't fit
            else:

                pieces.append(" ".join(current_words))

                # ---------------------------------
                # Word-level overlap
                # ---------------------------------
                overlap_words = []
                overlap_length = 0

                for previous_word in reversed(current_words):

                    extra_length = (
                        len(previous_word)
                        if not overlap_words
                        else len(previous_word) + 1
                    )

                    if overlap_length + extra_length <= overlap:

                        overlap_words.insert(0, previous_word)

                        overlap_length += extra_length

                    else:
                        break

                current_words = overlap_words + [word]

                current_length = len(" ".join(current_words))

        if current_words:

            pieces.append(" ".join(current_words))

        return pieces

    # -----------------------------------------
    # Process sentences
    # -----------------------------------------
    for sentence in sentences:

        # -------------------------------------
        # Very long sentence
        # -------------------------------------
        if len(sentence) > chunk_size:

            # Save previous sentences
            if current_sentences:

                chunks.append(build_chunk(current_sentences))

                available_overlap = chunk_size - len(sentence) - 1

                overlap_sentences = get_overlap_sentences(
                    current_sentences, available_overlap
                )

                current_sentences = overlap_sentences

                current_length = len(build_chunk(current_sentences))

            # Split long sentence by words
            long_sentence_chunks = split_long_sentence(sentence)

            # Save all pieces except final piece
            for piece in long_sentence_chunks[:-1]:

                chunks.append(piece)

            # Continue with final piece
            last_piece = long_sentence_chunks[-1]

            current_sentences = [last_piece]

            current_length = len(last_piece)

            continue

        # -------------------------------------
        # Normal sentence
        # -------------------------------------
        sentence_len = sentence_length(sentence)

        # No current chunk
        if not current_sentences:

            current_sentences = [sentence]

            current_length = sentence_len

            continue

        # -------------------------------------
        # Check if sentence fits
        # -------------------------------------
        combined_length = current_length + 1 + sentence_len

        if combined_length <= chunk_size:

            current_sentences.append(sentence)

            current_length = combined_length

            continue

        # -------------------------------------
        # Current chunk is full
        # -------------------------------------
        chunks.append(build_chunk(current_sentences))

        # -------------------------------------
        # Calculate available overlap space
        # -------------------------------------
        available_overlap = chunk_size - sentence_len - 1

        # -------------------------------------
        # Select overlap that can actually fit
        # -------------------------------------
        overlap_sentences = get_overlap_sentences(current_sentences, available_overlap)

        # -------------------------------------
        # Start next chunk with overlap
        # + new sentence
        # -------------------------------------
        current_sentences = overlap_sentences + [sentence]

        current_length = len(build_chunk(current_sentences))

        # -------------------------------------
        # Safety check
        # -------------------------------------
        if current_length > chunk_size:

            # Start with only the new sentence
            current_sentences = [sentence]

            current_length = sentence_len

    # -----------------------------------------
    # Save final chunk
    # -----------------------------------------
    if current_sentences:

        chunks.append(build_chunk(current_sentences))

    return chunks
