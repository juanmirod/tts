"""Tests for text chunking (tts/text_parser.py)."""

from tts.text_parser import chunk_text, parse_markdown, strip_vocal_tags


class TestChunkText:
    """Tests for chunk_text, including hard-splitting oversized paragraphs."""

    def test_short_text_is_single_chunk(self):
        assert chunk_text("hello world", 1500) == ["hello world"]

    def test_paragraphs_merged_up_to_limit(self):
        assert chunk_text("one\ntwo\nthree", 1500) == ["one\ntwo\nthree"]

    def test_split_on_paragraph_boundary(self):
        para_a, para_b = "a" * 100, "b" * 100
        assert chunk_text(para_a + "\n" + para_b, 150) == [para_a, para_b]

    def test_empty_lines_produce_no_empty_chunks(self):
        # Leading blank lines are kept inside the chunk (pre-existing
        # behavior); the point is no empty chunks are emitted.
        assert chunk_text("\n\nhello\n\n", 1500) == ["\n\nhello"]

    def test_oversized_paragraph_is_hard_split(self):
        para = " ".join(["word"] * 400)  # 1999 chars, over the 1500 limit
        chunks = chunk_text(para, 1500)
        assert all(len(c) <= 1500 for c in chunks)
        assert " ".join(chunks) == para

    def test_oversized_paragraph_after_normal_text(self):
        para = " ".join(["word"] * 400)
        chunks = chunk_text("intro\n" + para, 1500)
        assert chunks[0] == "intro"
        assert all(len(c) <= 1500 for c in chunks)
        assert " ".join(chunks[1:]) == para

    def test_tiny_limit_splits_on_words(self):
        assert chunk_text("aa bb cc dd ee", 5) == ["aa bb", "cc dd", "ee"]

    def test_single_long_word_is_cut_to_fit(self):
        assert chunk_text("x" * 2000, 1500) == ["x" * 1500, "x" * 500]

    def test_limit_respected_across_whole_chunk(self):
        text = "\n".join(["p" * 1000] * 5)
        chunks = chunk_text(text, 1500)
        assert all(len(c) <= 1500 for c in chunks)
        assert "\n".join(chunks) == text

    def test_default_4000_limit_still_works(self):
        chunks = chunk_text("a" * 4500, 4000)
        assert chunks == ["a" * 4000, "a" * 500]


class TestVocalTags:
    """parse_markdown keeps inline vocal tags only when asked to."""

    TEXT = "I have a secret. <short pause> Nobody knows. <sigh> Really."

    def test_tags_stripped_by_default(self):
        assert parse_markdown(self.TEXT) == "I have a secret. Nobody knows. Really."

    def test_tags_kept_and_spacing_intact(self):
        # The blockquote rule (`> +`) must not eat the space after a tag.
        assert parse_markdown(self.TEXT, keep_vocal_tags=True) == self.TEXT

    def test_html_still_stripped_when_keeping_tags(self):
        text = '<p class="x">Hi <b>there</b><br/> <laughs softly> ok</p>'
        assert parse_markdown(text, keep_vocal_tags=True) == "Hi there <laughs softly> ok"

    def test_markdown_still_cleaned_around_tags(self):
        text = "# Title\n**Bold** <gasp> and *italic*\n> quote <long pause>"
        assert parse_markdown(text, keep_vocal_tags=True) == \
            "Title\nBold <gasp> and italic\nquote <long pause>"

    def test_non_vocal_angle_brackets_still_stripped(self):
        text = "See <https://example.com> and <T> types <sigh>"
        assert parse_markdown(text, keep_vocal_tags=True) == "See and types <sigh>"


class TestStripVocalTags:
    def test_removes_tags_and_the_space_before_them(self):
        assert strip_vocal_tags("Great job! <laugh> You got it. <short pause>") == \
            "Great job! You got it."

    def test_leading_tag_leaves_text_intact(self):
        assert strip_vocal_tags("<gasp> Wow, really?") == " Wow, really?"

    def test_html_and_plain_text_untouched(self):
        text = "a < b and <b>bold</b> stay"
        assert strip_vocal_tags(text) == text
