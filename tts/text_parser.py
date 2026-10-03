import re

# Inline vocal tags understood by expressive TTS models (e.g. Gemini 3.8):
# lowercase words such as <sigh>, <laughs softly> or <short pause>.
VOCAL_TAG = re.compile(r'<([a-z]+)(?:[ -][a-z]+)*>')

# Tags with these names are HTML, never vocal tags, so they are always stripped.
HTML_ELEMENTS = {
    'a', 'abbr', 'address', 'area', 'article', 'aside', 'audio', 'b', 'base',
    'bdi', 'bdo', 'big', 'blockquote', 'body', 'br', 'button', 'canvas',
    'caption', 'center', 'cite', 'code', 'col', 'colgroup', 'data',
    'datalist', 'dd', 'del', 'details', 'dfn', 'dialog', 'div', 'dl', 'dt',
    'em', 'embed', 'fieldset', 'figcaption', 'figure', 'font', 'footer',
    'form', 'g', 'head', 'header', 'hgroup', 'hr', 'html', 'i', 'iframe',
    'img', 'input', 'ins', 'kbd', 'label', 'legend', 'li', 'link', 'main',
    'map', 'mark', 'menu', 'meta', 'meter', 'nav', 'noscript', 'object', 'ol',
    'optgroup', 'option', 'output', 'p', 'param', 'path', 'picture', 'pre',
    'progress', 'q', 'rp', 'rt', 'ruby', 's', 'samp', 'script', 'search',
    'section', 'select', 'slot', 'small', 'source', 'span', 'strike',
    'strong', 'style', 'sub', 'summary', 'sup', 'svg', 'table', 'tbody', 'td',
    'template', 'textarea', 'tfoot', 'th', 'thead', 'time', 'title', 'tr',
    'track', 'tt', 'u', 'ul', 'var', 'video', 'wbr',
}


def _stash_vocal_tags(text):
    """Swaps vocal tags for placeholders so the markdown/HTML cleanup below
    can't strip or mangle them. Returns (text, tags)."""
    tags = []

    def stash(match):
        if match.group(1) in HTML_ELEMENTS:
            return match.group(0)
        tags.append(match.group(0))
        return '\x00%d\x00' % (len(tags) - 1)

    return VOCAL_TAG.sub(stash, text), tags


def _restore_vocal_tags(text, tags):
    return re.sub(r'\x00(\d+)\x00', lambda m: tags[int(m.group(1))], text)


def strip_vocal_tags(text):
    """Removes inline vocal tags (and the spaces before them), for models
    that would read them aloud. HTML-looking tags are left alone."""
    def drop(match):
        return match.group(0) if match.group(2) in HTML_ELEMENTS else ''

    return re.sub(r'[ \t]*(' + VOCAL_TAG.pattern + ')', drop, text)


def parse_markdown(markdown, keep_vocal_tags=False):
    """Removes the frontmatter and html from a markdown string. With
    keep_vocal_tags, inline vocal tags like <sigh> survive (HTML doesn't)."""
    tags = []
    if keep_vocal_tags:
        markdown, tags = _stash_vocal_tags(markdown)

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
    if tags:
        text = _restore_vocal_tags(text, tags)
    return text


def _split_words(text, max_length):
    """Hard-splits text on word boundaries so no piece exceeds max_length
    (absurdly long words are cut to fit)."""
    pieces = []
    current = ""
    for word in text.split(' '):
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


def _split_long_paragraph(paragraph, max_length):
    """Splits a single paragraph longer than max_length into pieces of at
    most max_length chars. Pieces end at sentence boundaries when possible;
    a sentence that is longer than the limit by itself is split on word
    boundaries."""
    pieces = []
    current = ""
    for sentence in re.split(r'(?<=[.!?])\s+', paragraph):
        if len(sentence) > max_length:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_split_words(sentence, max_length))
        elif not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_length:
            current += ' ' + sentence
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


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
