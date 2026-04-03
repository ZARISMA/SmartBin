"""Tests for smartwaste/prompt.py — validate the Gemini prompt string."""

from smartwaste.prompt import PROMPT


class TestPromptStructure:
    def test_is_nonempty_string(self):
        assert isinstance(PROMPT, str) and len(PROMPT) > 0

    def test_no_leading_trailing_whitespace(self):
        # prompt.py ends with .strip(), so there should be none
        assert PROMPT == PROMPT.strip()

    def test_mentions_json_output(self):
        assert "JSON" in PROMPT or "json" in PROMPT


class TestPromptCategories:
    def test_contains_all_valid_categories(self):
        from smartwaste.config import VALID_CLASSES
        for category in VALID_CLASSES:
            assert category in PROMPT, f"Missing category '{category}' in prompt"

    def test_mentions_empty_category_rule(self):
        assert "Empty" in PROMPT

    def test_mentions_other_fallback(self):
        assert "Other" in PROMPT


class TestPromptJsonKeys:
    def test_category_key_present(self):
        assert '"category"' in PROMPT

    def test_description_key_present(self):
        assert '"description"' in PROMPT

    def test_brand_product_key_present(self):
        assert '"brand_product"' in PROMPT


class TestPromptDualCamera:
    def test_mentions_two_cameras(self):
        assert "Camera A" in PROMPT or "LEFT" in PROMPT
        assert "Camera B" in PROMPT or "RIGHT" in PROMPT

    def test_mentions_side_by_side(self):
        lower = PROMPT.lower()
        assert "side" in lower or "left" in lower


class TestPromptArmenianBrands:
    def test_mentions_jermuk(self):
        assert "Jermuk" in PROMPT

    def test_mentions_bjni(self):
        assert "Bjni" in PROMPT

    def test_mentions_boom(self):
        assert "BOOM" in PROMPT
