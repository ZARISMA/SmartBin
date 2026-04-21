"""
Tests for smartwaste/classifier.py.

_extract_json  — tested directly (no network calls needed).
classify()     — tested with a mocked Gemini client.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from smartwaste.classifier import _extract_json
from smartwaste.config import VALID_CLASSES
from smartwaste.state import AppState


# ─────────────────────────────────────────────────────────────────────────────
# _extract_json
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractJsonCleanInput:
    def test_plain_json_object(self):
        t = '{"category": "Plastic", "description": "bottle", "brand_product": "Coca-Cola"}'
        assert _extract_json(t)["category"] == "Plastic"

    def test_all_three_keys_returned(self):
        t = '{"category": "Paper", "description": "newspaper", "brand_product": "None"}'
        result = _extract_json(t)
        assert set(result.keys()) == {"category", "description", "brand_product"}

    def test_whitespace_padded(self):
        t = '   {"category": "Glass"}   '
        assert _extract_json(t)["category"] == "Glass"

    def test_newlines_around_json(self):
        t = '\n\n{"category": "Aluminum"}\n\n'
        assert _extract_json(t)["category"] == "Aluminum"

    def test_empty_object(self):
        assert _extract_json("{}") == {}

    def test_nested_value(self):
        t = '{"category": "Other", "description": "misc", "brand_product": "N/A", "meta": {"x": 1}}'
        assert _extract_json(t)["category"] == "Other"

    def test_unicode_description(self):
        t = '{"category": "Organic", "description": "Ջերմուկ շիշ", "brand_product": "Jermuk"}'
        assert _extract_json(t)["description"] == "Ջերմուկ շիշ"

    def test_unicode_brand(self):
        t = '{"category": "Plastic", "description": "bottle", "brand_product": "ԲՈՒՄ"}'
        assert _extract_json(t)["brand_product"] == "ԲՈՒՄ"

    def test_numeric_like_string_value(self):
        t = '{"category": "Other", "description": "123", "brand_product": "Unknown"}'
        assert _extract_json(t)["description"] == "123"

    def test_all_valid_categories_parse(self):
        for cat in VALID_CLASSES:
            t = f'{{"category": "{cat}", "description": "x", "brand_product": "y"}}'
            assert _extract_json(t)["category"] == cat


class TestExtractJsonMarkdownFences:
    def test_json_code_fence(self):
        t = '```json\n{"category": "Aluminum"}\n```'
        assert _extract_json(t)["category"] == "Aluminum"

    def test_plain_code_fence(self):
        t = '```\n{"category": "Glass"}\n```'
        assert _extract_json(t)["category"] == "Glass"

    def test_fence_with_extra_blank_lines(self):
        t = '```json\n\n  {"category": "Paper"}  \n\n```'
        assert _extract_json(t)["category"] == "Paper"

    def test_fence_with_full_response(self):
        t = '```json\n{"category": "Organic", "description": "food", "brand_product": "Unknown"}\n```'
        result = _extract_json(t)
        assert result["category"] == "Organic"


class TestExtractJsonGarbageSurround:
    def test_garbage_prefix(self):
        t = 'Sure! Here is the result: {"category": "Plastic"}'
        assert _extract_json(t)["category"] == "Plastic"

    def test_garbage_suffix(self):
        t = '{"category": "Glass"} — that is my answer.'
        assert _extract_json(t)["category"] == "Glass"

    def test_garbage_both_sides(self):
        t = 'The waste is {"category": "Organic", "description": "food"} end.'
        assert _extract_json(t)["category"] == "Organic"

    def test_sentence_before_and_after(self):
        t = 'I detected: {"category": "Aluminum", "brand_product": "BOOM"} Done.'
        result = _extract_json(t)
        assert result["brand_product"] == "BOOM"

    def test_multiline_garbage(self):
        t = 'Analysis:\nLine 2\n{"category": "Paper"}\nEnd.'
        assert _extract_json(t)["category"] == "Paper"


class TestExtractJsonErrors:
    def test_empty_string_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")

    def test_whitespace_only_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("   ")

    def test_plain_text_no_braces_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("just plain text with no json")

    def test_missing_closing_brace_raises(self):
        with pytest.raises((json.JSONDecodeError, Exception)):
            _extract_json('{"category": "Plastic"')

    def test_only_opening_brace_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("{")

    def test_only_closing_brace_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("}")

    def test_none_like_empty_raises(self):
        # (text or "") handles falsy input
        with pytest.raises(json.JSONDecodeError):
            _extract_json("")

    def test_array_input_returns_list(self):
        # json.loads succeeds on a valid JSON array; _extract_json does not guard
        # against non-dict results — callers must handle this
        result = _extract_json('["a", "b"]')
        assert result == ["a", "b"]

    def test_empty_fence_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json("```json\n\n```")


# ─────────────────────────────────────────────────────────────────────────────
# classify() — mocked Gemini client
# ─────────────────────────────────────────────────────────────────────────────


def _mock_response(json_text: str) -> MagicMock:
    r = MagicMock()
    r.text = json_text
    return r


def _run_classify(response_text: str, state=None):
    """Helper: run classify() with a mocked client."""
    import smartwaste.classifier as clf

    if state is None:
        state = AppState()
        state.start_classify()

    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with (
        patch.object(clf, "client") as mock_client,
        patch("smartwaste.classifier.save_entry") as mock_save,
    ):
        mock_client.models.generate_content.return_value = _mock_response(response_text)
        from smartwaste.classifier import classify

        classify(b"fake_bytes", frame, state)
        return state, mock_save


class TestClassifyHappyPath:
    def test_valid_category_sets_label(self):
        state, _ = _run_classify(
            '{"category": "Plastic", "description": "bottle", "brand_product": "Coke"}'
        )
        label, _, _ = state.get_display()
        assert label == "Plastic"

    def test_detail_contains_brand_and_description(self):
        state, _ = _run_classify(
            '{"category": "Glass", "description": "jar", "brand_product": "Jermuk"}'
        )
        _, detail, _ = state.get_display()
        assert "Jermuk" in detail
        assert "jar" in detail

    def test_save_entry_called_for_non_empty(self):
        _, mock_save = _run_classify(
            '{"category": "Plastic", "description": "x", "brand_product": "y"}'
        )
        assert mock_save.called

    def test_save_entry_not_called_for_empty(self):
        _, mock_save = _run_classify(
            '{"category": "Empty", "description": "N/A", "brand_product": "Unknown"}'
        )
        assert not mock_save.called

    def test_finish_classify_called_on_success(self):
        state = AppState()
        state.start_classify()
        _run_classify(
            '{"category": "Paper", "description": "d", "brand_product": "b"}', state=state
        )
        # After classify(), finish_classify() was called, so start_classify should succeed again
        assert state.start_classify() is True

    def test_all_valid_categories_accepted(self):
        for cat in VALID_CLASSES:
            state, _ = _run_classify(
                f'{{"category": "{cat}", "description": "x", "brand_product": "y"}}'
            )
            label, _, _ = state.get_display()
            if cat != "Empty":
                assert label == cat


class TestClassifyCategoryNormalization:
    def test_lowercase_category_capitalised(self):
        state, _ = _run_classify(
            '{"category": "plastic", "description": "x", "brand_product": "y"}'
        )
        label, _, _ = state.get_display()
        assert label == "Plastic"

    def test_invalid_category_becomes_other(self):
        state, _ = _run_classify(
            '{"category": "Garbage", "description": "x", "brand_product": "y"}'
        )
        label, _, _ = state.get_display()
        assert label == "Other"

    def test_unknown_category_becomes_other(self):
        state, _ = _run_classify(
            '{"category": "Unicorn", "description": "x", "brand_product": "y"}'
        )
        label, _, _ = state.get_display()
        assert label == "Other"

    def test_missing_category_key_becomes_other(self):
        state, _ = _run_classify('{"description": "x", "brand_product": "y"}')
        label, _, _ = state.get_display()
        assert label == "Other"


class TestClassifyErrorHandling:
    def test_quota_error_sets_quota_label(self):
        import smartwaste.classifier as clf

        state = AppState()
        state.start_classify()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.object(clf, "client") as mock_client:
            mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED")
            from smartwaste.classifier import classify

            classify(b"bytes", frame, state)

        label, _, _ = state.get_display()
        assert "429" in label or "Quota" in label

    def test_generic_error_sets_error_label(self):
        import smartwaste.classifier as clf

        state = AppState()
        state.start_classify()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.object(clf, "client") as mock_client:
            mock_client.models.generate_content.side_effect = Exception("Network timeout")
            from smartwaste.classifier import classify

            classify(b"bytes", frame, state)

        label, _, _ = state.get_display()
        assert label == "Error"

    def test_finish_classify_called_on_error(self):
        import smartwaste.classifier as clf

        state = AppState()
        state.start_classify()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.object(clf, "client") as mock_client:
            mock_client.models.generate_content.side_effect = Exception("boom")
            from smartwaste.classifier import classify

            classify(b"bytes", frame, state)

        assert state.start_classify() is True  # finish_classify was called

    def test_bad_json_response_does_not_crash(self):
        import smartwaste.classifier as clf

        state = AppState()
        state.start_classify()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)

        with patch.object(clf, "client") as mock_client, patch("smartwaste.classifier.save_entry"):
            mock_client.models.generate_content.return_value = _mock_response("not json at all")
            from smartwaste.classifier import classify

            classify(b"bytes", frame, state)

        # Should either set Error status or handle gracefully
        assert state.start_classify() is True  # finish_classify was called
