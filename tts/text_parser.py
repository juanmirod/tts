import re


def parse_markdown(markdown):
    """Removes the frontmatter and html from a markdown string."""
    patterns_without_group = [
        r'---.*---',          # Frontmatter
        r'<figure>.*<\/figure>',  # Figures
        r'<[^>]+>',           # HTML tags
        r'\!\[.*?\]\(.*?\)',  # Images
        r'\# *',               # Headers
        r'\- +',               # Horizontal rules and lists
        r'\> +',               # Blockquotes
        r'\[\^\d\]: ',           # Footnotes
    ]

    text = markdown
    for pattern in patterns_without_group:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    # Remove common Markdown characters
    patterns = [
        r'\[(.*?)\]\(.*?\)',  # Links
        r'\*\*(.*?)\*\*',     # Bold
        r'\*(.*?)\*',         # Italic
        r'\~\~(.*?)\~\~',     # Strikethrough
        r'\`(.*?)\`',         # Inline code
    ]

    for pattern in patterns:
        text = re.sub(pattern, r'\1', text)

    # Convert multiple spaces to single space
    text = re.sub(r' {2,}', ' ', text)
    # Convert multiple newlines to single newline
    text = re.sub(r'\n{2,}', '\n', text)

    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def _split_long_paragraph(paragraph, max_length):
    """Hard-splits a single paragraph longer than max_length on word
    boundaries so no piece exceeds max_length (absurdly long words are
    cut to fit). Returns pieces that re-join with ' ' to the original."""
    pieces = []
    current = ""
    for word in paragraph.split(' '):
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_length:
            current += ' ' + word
        else:
            pieces.append(current)
            current = word
    if current:
        pieces.append(current)
    # Cut single tokens longer than the limit (very rare).
    result = []
    for piece in pieces:
        if len(piece) > max_length:
            result.extend(piece[i:i + max_length]
                          for i in range(0, len(piece), max_length))
        else:
            result.append(piece)
    return result


def chunk_text(text, max_length):
    """Splits text into chunks of at most max_length chars. Normal chunks
    keep whole paragraphs; a single paragraph longer than max_length is
    hard-split on word boundaries so it never exceeds the limit."""
    chunks = []
    current_chunk = ""

    for paragraph in text.split('\n'):
        if len(current_chunk) + len(paragraph) + 1 <= max_length:  # +1 for the newline
            current_chunk += paragraph + '\n'
        elif len(paragraph) > max_length:
            # Flush what we have, then hard-split the oversized paragraph.
            chunks.append(current_chunk.rstrip('\n'))
            current_chunk = ""
            chunks.extend(_split_long_paragraph(paragraph, max_length))
        else:
            # remove trailing '\n' from the chunk
            chunks.append(current_chunk.rstrip('\n'))
            current_chunk = paragraph + '\n'

    # add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.rstrip('\n'))

    return [chunk for chunk in chunks if chunk]
