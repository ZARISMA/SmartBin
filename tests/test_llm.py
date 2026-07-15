"""Tests for hexabin/llm.py — result parsing, backends, cascade, factory."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import hexabin.llm as llm
from hexabin.config import LMSTUDIO_MODEL
from hexabin.llm import (
    CascadeBackend,
    CircuitOpenError,
    ClassificationResult,
    GeminiBackend,
    LLMError,
    LMStudioBackend,
    build_backend,
    parse_result,
)
from hexabin.prompt import PROMPT


@pytest.fixture(autouse=True)
def _reset_llm_state():
    """Reset the lazy Gemini client and circuit-breaker between tests."""
    llm._client = None
    llm._record_success()
    yield
    llm._client = None
    llm._record_success()


def _result(category="Plastic", confidence=0.9, backend="lmstudio"):
    return ClassificationResult(
        category=category,
        description="d",
        brand_product="b",
        confidence=confidence,
        backend=backend,
    )


# ─────────────────────────────────────────────────────────────────────────────
# parse_result
# ─────────────────────────────────────────────────────────────────────────────


class TestParseResult:
    def test_parses_all_fields(self):
        raw = '{"category": "Glass", "description": "jar", "brand_product": "Jermuk", "confidence": 85}'
        r = parse_result(raw, "lmstudio")
        assert r.category == "Glass"
        assert r.description == "jar"
        assert r.brand_product == "Jermuk"
        assert r.confidence == pytest.approx(0.85)
        assert r.backend == "lmstudio"
        assert r.escalated is False

    def test_percent_confidence_normalized(self):
        r = parse_result('{"category": "Paper", "confidence": 40}', "gemini")
        assert r.confidence == pytest.approx(0.40)

    def test_fractional_confidence_kept(self):
        r = parse_result('{"category": "Paper", "confidence": 0.4}', "gemini")
        assert r.confidence == pytest.approx(0.40)

    def test_numeric_string_confidence(self):
        r = parse_result('{"category": "Paper", "confidence": "72"}', "gemini")
        assert r.confidence == pytest.approx(0.72)

    def test_missing_confidence_is_none(self):
        r = parse_result('{"category": "Paper"}', "gemini")
        assert r.confidence is None

    def test_garbage_confidence_is_none(self):
        r = parse_result('{"category": "Paper", "confidence": "very sure"}', "gemini")
        assert r.confidence is None

    def test_bool_confidence_is_none(self):
        r = parse_result('{"category": "Paper", "confidence": true}', "gemini")
        assert r.confidence is None

    def test_oversized_confidence_clamped(self):
        r = parse_result('{"category": "Paper", "confidence": 250}', "gemini")
        assert r.confidence == pytest.approx(1.0)

    def test_invalid_category_becomes_other(self):
        r = parse_result('{"category": "Unicorn"}', "gemini")
        assert r.category == "Other"

    def test_lowercase_category_capitalised(self):
        r = parse_result('{"category": "plastic"}', "gemini")
        assert r.category == "Plastic"

    def test_non_dict_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_result('["a", "b"]', "gemini")

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_result("nothing here", "gemini")


# ─────────────────────────────────────────────────────────────────────────────
# LMStudioBackend
# ─────────────────────────────────────────────────────────────────────────────


def _lms_body(content: str) -> bytes:
    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


class TestLMStudioBackend:
    def test_happy_path(self):
        content = '{"category": "Aluminum", "description": "can", "brand_product": "BOOM", "confidence": 90}'
        with patch("hexabin.llm.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = _lms_body(content)
            r = LMStudioBackend().classify(b"img")
        assert r.category == "Aluminum"
        assert r.backend == "lmstudio"
        assert r.confidence == pytest.approx(0.90)

    def test_request_payload_shape(self):
        with patch("hexabin.llm.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = _lms_body(
                '{"category": "Glass"}'
            )
            LMStudioBackend().classify(b"img-bytes")
            req = mock_open.call_args[0][0]

        assert req.full_url.endswith("/chat/completions")
        payload = json.loads(req.data)
        assert payload["model"] == LMSTUDIO_MODEL
        assert payload["temperature"] == 0
        assert payload["stream"] is False
        content = payload["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == PROMPT

    def test_connection_error_raises_llmerror(self):
        with patch(
            "hexabin.llm.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(LLMError):
                LMStudioBackend().classify(b"img")

    def test_http_error_raises_llmerror_with_code(self):
        import io

        err = urllib.error.HTTPError("http://x/v1/chat/completions", 404, "nf", {}, io.BytesIO(b""))
        with patch("hexabin.llm.urllib.request.urlopen", side_effect=err):
            with pytest.raises(LLMError, match="404"):
                LMStudioBackend().classify(b"img")

    def test_unparseable_content_raises_llmerror(self):
        with patch("hexabin.llm.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = _lms_body(
                "sorry, no json"
            )
            with pytest.raises(LLMError):
                LMStudioBackend().classify(b"img")

    def test_unexpected_response_shape_raises_llmerror(self):
        with patch("hexabin.llm.urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = b'{"oops": 1}'
            with pytest.raises(LLMError):
                LMStudioBackend().classify(b"img")


# ─────────────────────────────────────────────────────────────────────────────
# GeminiBackend — lazy client + circuit breaker
# ─────────────────────────────────────────────────────────────────────────────


class TestGeminiBackend:
    def test_construction_never_needs_a_key(self):
        GeminiBackend()  # must not raise — the client is built lazily

    def test_missing_key_raises_at_call_time(self):
        with patch.object(llm.settings, "gemini_api_key", ""):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                llm._get_client()

    def test_classify_parses_response(self):
        raw = '{"category": "Glass", "description": "jar", "brand_product": "Jermuk", "confidence": 80}'
        with patch("hexabin.llm._call_gemini", return_value=raw):
            r = GeminiBackend().classify(b"img")
        assert r.category == "Glass"
        assert r.backend == "gemini"
        assert r.confidence == pytest.approx(0.80)

    def test_circuit_opens_after_repeated_failures(self):
        with patch("hexabin.llm._call_gemini", side_effect=Exception("boom")):
            for _ in range(llm.settings.cb_failure_threshold):
                with pytest.raises(Exception):
                    GeminiBackend().classify(b"img")
            with pytest.raises(CircuitOpenError):
                GeminiBackend().classify(b"img")

    def test_success_resets_failure_count(self):
        raw = '{"category": "Paper"}'
        with patch("hexabin.llm._call_gemini", side_effect=Exception("boom")):
            with pytest.raises(Exception):
                GeminiBackend().classify(b"img")
        with patch("hexabin.llm._call_gemini", return_value=raw):
            assert GeminiBackend().classify(b"img").category == "Paper"
        assert llm._cb_failures == 0


# ─────────────────────────────────────────────────────────────────────────────
# CascadeBackend
# ─────────────────────────────────────────────────────────────────────────────


def _stub_backend(name="stub", result=None, error=None):
    b = MagicMock()
    b.name = name
    if error is not None:
        b.classify.side_effect = error
    else:
        b.classify.return_value = result
    return b


class TestCascadeBackend:
    def test_high_confidence_stays_local(self):
        primary = _stub_backend("lmstudio", result=_result(confidence=0.95))
        fallback = _stub_backend("gemini")
        r = CascadeBackend(primary, fallback, threshold=0.70).classify(b"img")
        assert r.backend == "lmstudio"
        assert r.escalated is False
        assert not fallback.classify.called

    def test_boundary_confidence_stays_local(self):
        primary = _stub_backend("lmstudio", result=_result(confidence=0.70))
        fallback = _stub_backend("gemini")
        r = CascadeBackend(primary, fallback, threshold=0.70).classify(b"img")
        assert not fallback.classify.called
        assert r.backend == "lmstudio"

    def test_low_confidence_escalates(self):
        primary = _stub_backend("lmstudio", result=_result(category="Other", confidence=0.30))
        fallback = _stub_backend("gemini", result=_result(category="Glass", backend="gemini"))
        r = CascadeBackend(primary, fallback, threshold=0.70).classify(b"img")
        assert fallback.classify.called
        assert r.category == "Glass"
        assert r.backend == "gemini"
        assert r.escalated is True

    def test_missing_confidence_escalates(self):
        primary = _stub_backend("lmstudio", result=_result(confidence=None))
        fallback = _stub_backend("gemini", result=_result(backend="gemini"))
        r = CascadeBackend(primary, fallback, threshold=0.70).classify(b"img")
        assert fallback.classify.called
        assert r.escalated is True

    def test_primary_failure_escalates(self):
        primary = _stub_backend("lmstudio", error=LLMError("LM Studio down"))
        fallback = _stub_backend("gemini", result=_result(backend="gemini"))
        r = CascadeBackend(primary, fallback, threshold=0.70).classify(b"img")
        assert r.backend == "gemini"
        assert r.escalated is True

    def test_fallback_failure_keeps_low_confidence_primary(self):
        primary_result = _result(confidence=0.20)
        primary = _stub_backend("lmstudio", result=primary_result)
        fallback = _stub_backend("gemini", error=LLMError("quota"))
        r = CascadeBackend(primary, fallback, threshold=0.70).classify(b"img")
        assert r is primary_result
        assert r.escalated is False

    def test_both_failing_raises_llmerror(self):
        primary = _stub_backend("lmstudio", error=LLMError("down"))
        fallback = _stub_backend("gemini", error=CircuitOpenError("open"))
        with pytest.raises(LLMError):
            CascadeBackend(primary, fallback, threshold=0.70).classify(b"img")


# ─────────────────────────────────────────────────────────────────────────────
# build_backend factory
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildBackend:
    def test_gemini(self):
        assert isinstance(build_backend("gemini"), GeminiBackend)

    def test_lmstudio(self):
        assert isinstance(build_backend("lmstudio"), LMStudioBackend)

    def test_cascade_wiring(self):
        b = build_backend("cascade")
        assert isinstance(b, CascadeBackend)
        assert isinstance(b.primary, LMStudioBackend)
        assert isinstance(b.fallback, GeminiBackend)
        assert b.threshold == pytest.approx(0.70)

    def test_default_comes_from_config(self):
        assert isinstance(build_backend(), GeminiBackend)  # config default

    def test_case_insensitive(self):
        assert isinstance(build_backend("LMStudio"), LMStudioBackend)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_backend("chatgpt")
