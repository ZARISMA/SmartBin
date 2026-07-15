"""
hexabin/llm.py — pluggable LLM classification backends.

Backends (selected by HEXABIN_LLM_BACKEND):

* ``gemini``   — Google Gemini cloud API. Carries over the original resilience
  features: tenacity retry with exponential backoff (quota errors are not
  retried) and a circuit breaker that pauses calls after repeated failures.
* ``lmstudio`` — local model served by LM Studio (or any OpenAI-compatible
  server: vLLM, llama.cpp, Ollama's /v1 endpoint) via /chat/completions.
* ``cascade``  — LM Studio first; escalate the same image to Gemini when the
  local model fails, reports no confidence, or its confidence is below
  HEXABIN_CONFIDENCE_THRESHOLD.

All backends return a ClassificationResult; they never touch AppState or the
database — callers (classifier.py, web.py) own persistence and UI.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    CONFIDENCE_THRESHOLD,
    LLM_BACKEND,
    LMSTUDIO_MAX_TOKENS,
    LMSTUDIO_MODEL,
    LMSTUDIO_TIMEOUT,
    LMSTUDIO_URL,
    MODEL_NAME,
    VALID_CLASSES,
)
from .log_setup import get_logger
from .prompt import PROMPT
from .settings import settings

if TYPE_CHECKING:
    from google import genai

logger = get_logger()


class LLMError(Exception):
    """A classification backend failed to produce a result."""


class CircuitOpenError(LLMError):
    """Gemini circuit breaker is open — calls are paused."""


@dataclass
class ClassificationResult:
    """Normalized output of any backend."""

    category: str  # always one of VALID_CLASSES
    description: str
    brand_product: str
    confidence: float | None  # 0.0-1.0, None if the model didn't report one
    backend: str  # "gemini" | "lmstudio"
    escalated: bool = False  # True when cascade fell through to the fallback
    raw: str = ""  # raw model text (debugging)


class LLMBackend(Protocol):
    name: str

    def classify(self, img_bytes: bytes) -> ClassificationResult: ...


# ── JSON extraction / result parsing ──────────────────────────────────────────


def extract_json(text: str) -> dict:  # type: ignore[type-arg]
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```", "").strip()
    try:
        result: dict = json.loads(t)  # type: ignore[assignment]
        return result
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        result = json.loads(t[start : end + 1])  # type: ignore[assignment]
        return result
    raise json.JSONDecodeError("No JSON object found", t, 0)


def _parse_confidence(value: object) -> float | None:
    """Tolerant confidence parse: int/float/numeric-str; >1 treated as percent;
    clamped to [0, 1]; anything else → None (never fabricate certainty)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        conf = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if conf > 1.0:
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


def parse_result(raw: str, backend: str) -> ClassificationResult:
    """Parse a raw model response into a ClassificationResult."""
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("Expected a JSON object", raw, 0)

    label = str(data.get("category", "Other")).strip().capitalize()
    if label not in VALID_CLASSES:
        label = "Other"

    return ClassificationResult(
        category=label,
        description=str(data.get("description", "N/A")).strip(),
        brand_product=str(data.get("brand_product", "Unknown")).strip(),
        confidence=_parse_confidence(data.get("confidence")),
        backend=backend,
        raw=raw,
    )


# ── Gemini circuit breaker ─────────────────────────────────────────────────────

_cb_lock: threading.Lock = threading.Lock()
_cb_failures: int = 0
_cb_open_until: float = 0.0


def circuit_is_open() -> bool:
    """Return True when the Gemini circuit is open (calls should be skipped)."""
    global _cb_open_until
    with _cb_lock:
        if _cb_open_until and time.time() < _cb_open_until:
            return True
        if _cb_open_until:
            _cb_open_until = 0.0  # recovery window elapsed → half-open, allow one try
        return False


def _record_success() -> None:
    global _cb_failures, _cb_open_until
    with _cb_lock:
        _cb_failures = 0
        _cb_open_until = 0.0


def _record_failure() -> bool:
    """Record a final failure (after all retries). Returns True if circuit just opened."""
    global _cb_failures, _cb_open_until
    with _cb_lock:
        _cb_failures += 1
        if _cb_failures >= settings.cb_failure_threshold:
            _cb_open_until = time.time() + settings.cb_recovery_sec
            logger.error(
                "Circuit breaker OPENED after %d consecutive failures — "
                "Gemini calls paused for %.0f s.",
                _cb_failures,
                settings.cb_recovery_sec,
            )
            _cb_failures = 0  # reset so next window gets a fresh count
            return True
        return False


# ── Gemini backend ─────────────────────────────────────────────────────────────

_QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "Quota exceeded", "limit: 0")

_client: "genai.Client | None" = None
_client_lock = threading.Lock()


def is_quota_error(msg: str) -> bool:
    return any(m in msg for m in _QUOTA_MARKERS)


def _is_retryable(exc: BaseException) -> bool:
    """Quota errors won't recover within seconds — don't retry them."""
    return not is_quota_error(str(exc))


def _build_client() -> "genai.Client":
    from google import genai  # imported lazily — not needed on LM Studio-only hosts

    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "  Windows : $env:GEMINI_API_KEY='your_key'\n"
            "  Linux   : export GEMINI_API_KEY='your_key'\n"
            "  Or add GEMINI_API_KEY=your_key to a .env file in the project root."
        )
    return genai.Client(api_key=settings.gemini_api_key)


def _get_client() -> "genai.Client":
    """Build the Gemini client on first use (a server without a Gemini key can
    still import this module and run other backends)."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _build_client()
    return _client


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(settings.api_retry_attempts),
    wait=wait_exponential(
        multiplier=1,
        min=settings.api_retry_min_wait,
        max=settings.api_retry_max_wait,
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_gemini(img_bytes: bytes) -> str:
    """Call Gemini and return the raw text response. Tenacity handles retries."""
    resp = _get_client().models.generate_content(
        model=MODEL_NAME,
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64.b64encode(img_bytes).decode(),
                        }
                    },
                    {"text": PROMPT},
                ],
            }
        ],
    )
    return (resp.text or "").strip()


class GeminiBackend:
    """Google Gemini cloud API with retry + circuit breaker."""

    name = "gemini"

    def classify(self, img_bytes: bytes) -> ClassificationResult:
        if circuit_is_open():
            raise CircuitOpenError(
                f"Circuit open after repeated failures — "
                f"retrying in up to {settings.cb_recovery_sec:.0f}s"
            )
        logger.info("Gemini request → model=%s  bytes=%d", MODEL_NAME, len(img_bytes))
        try:
            raw = _call_gemini(img_bytes)
            _record_success()
            logger.info("Gemini raw response: %s", raw)
            return parse_result(raw, self.name)
        except Exception:
            _record_failure()
            raise


# ── LM Studio backend (OpenAI-compatible chat completions) ─────────────────────


class LMStudioBackend:
    """Local model served by LM Studio (or any OpenAI-compatible server).

    LM Studio's server is unauthenticated — expose it only on a trusted LAN.
    """

    name = "lmstudio"

    def classify(self, img_bytes: bytes) -> ClassificationResult:
        payload = {
            "model": LMSTUDIO_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,"
                                + base64.b64encode(img_bytes).decode("ascii")
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "temperature": 0,
            # Generous: reasoning models (gemma-4 etc.) spend completion tokens
            # on reasoning_content BEFORE the JSON answer.
            "max_tokens": LMSTUDIO_MAX_TOKENS,
            "stream": False,
        }
        url = f"{LMSTUDIO_URL.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        logger.info(
            "LM Studio request → model=%s  url=%s  bytes=%d", LMSTUDIO_MODEL, url, len(img_bytes)
        )
        try:
            with urllib.request.urlopen(req, timeout=LMSTUDIO_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise LLMError(f"LM Studio returned HTTP {e.code}: {detail}") from e
        except Exception as e:
            raise LLMError(f"LM Studio request failed: {e}") from e

        try:
            choice = body["choices"][0]
            raw = str(choice["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected LM Studio response shape: {e}") from e

        if not raw:
            finish = choice.get("finish_reason", "?")
            raise LLMError(
                f"LM Studio returned empty content (finish_reason={finish!r}) — "
                f"if 'length', raise HEXABIN_LMSTUDIO_MAX_TOKENS."
            )

        logger.info("LM Studio raw response: %s", raw)
        try:
            return parse_result(raw, self.name)
        except Exception as e:
            raise LLMError(f"LM Studio returned unparseable JSON: {raw[:200]}") from e


# ── Cascade backend ────────────────────────────────────────────────────────────


class CascadeBackend:
    """Local model first; escalate to the cloud fallback when the local model
    fails, reports no confidence, or is below the confidence threshold."""

    name = "cascade"

    def __init__(
        self,
        primary: LLMBackend,
        fallback: LLMBackend,
        threshold: float | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.threshold = CONFIDENCE_THRESHOLD if threshold is None else threshold

    def classify(self, img_bytes: bytes) -> ClassificationResult:
        primary_result: ClassificationResult | None = None
        try:
            primary_result = self.primary.classify(img_bytes)
        except Exception as exc:
            logger.warning(
                "Cascade: %s failed (%s) — escalating to %s.",
                self.primary.name,
                exc,
                self.fallback.name,
            )
        else:
            conf = primary_result.confidence
            if conf is not None and conf >= self.threshold:
                return primary_result
            logger.info(
                "Cascade: %s confidence %s below %.2f — escalating to %s.",
                self.primary.name,
                "n/a" if conf is None else f"{conf:.2f}",
                self.threshold,
                self.fallback.name,
            )

        try:
            result = self.fallback.classify(img_bytes)
            result.escalated = True
            return result
        except Exception as exc:
            if primary_result is not None:
                # Degraded beats erroring: keep the parseable low-confidence result.
                logger.warning(
                    "Cascade: fallback %s failed (%s) — keeping low-confidence %s result.",
                    self.fallback.name,
                    exc,
                    self.primary.name,
                )
                return primary_result
            raise LLMError(f"Both LLM backends failed; last error: {exc}") from exc


# ── Factory ────────────────────────────────────────────────────────────────────


def build_backend(name: str | None = None) -> LLMBackend:
    """Build the backend named by *name* (default: HEXABIN_LLM_BACKEND)."""
    resolved = (name or LLM_BACKEND).strip().lower()
    if resolved == "gemini":
        return GeminiBackend()
    if resolved == "lmstudio":
        return LMStudioBackend()
    if resolved == "cascade":
        return CascadeBackend(LMStudioBackend(), GeminiBackend())
    raise ValueError(
        f"Unknown LLM backend {resolved!r} — expected 'gemini', 'lmstudio', or 'cascade'."
    )
