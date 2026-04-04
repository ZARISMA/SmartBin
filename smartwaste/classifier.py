"""
smartwaste/classifier.py — Gemini classification worker.

Resilience features
───────────────────
* Retry with exponential backoff (tenacity): up to api_retry_attempts tries,
  waiting api_retry_min_wait … api_retry_max_wait seconds between each.
  Quota errors (HTTP 429) are NOT retried — they won't recover in seconds.

* Circuit breaker: after cb_failure_threshold consecutive final failures the
  circuit opens and Gemini calls are blocked for cb_recovery_sec seconds.
  This prevents hammering a down API and provides a clear user-visible status.
  The circuit resets automatically when the recovery window elapses.

All thresholds are read from smartwaste.settings so they can be overridden
via env vars or .env without touching source code.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time

from google import genai
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import MODEL_NAME, VALID_CLASSES
from .dataset import save_entry
from .log_setup import ERR_JSON_FILE, get_logger
from .prompt import PROMPT
from .settings import settings

logger = get_logger()

# ── Gemini client ──────────────────────────────────────────────────────────────

def _build_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "  Windows : $env:GEMINI_API_KEY='your_key'\n"
            "  Linux   : export GEMINI_API_KEY='your_key'\n"
            "  Or add GEMINI_API_KEY=your_key to a .env file in the project root."
        )
    return genai.Client(api_key=settings.gemini_api_key)


client = _build_client()

# ── Circuit breaker ────────────────────────────────────────────────────────────

_cb_lock:       threading.Lock = threading.Lock()
_cb_failures:   int            = 0
_cb_open_until: float          = 0.0


def _circuit_is_open() -> bool:
    """Return True when the circuit is open (API calls should be skipped)."""
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
        _cb_failures   = 0
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
            _cb_failures = 0   # reset so next window gets a fresh count
            return True
        return False


# ── JSON extraction ────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:  # type: ignore[type-arg]
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


# ── Gemini API call with retry ─────────────────────────────────────────────────

_QUOTA_MARKERS = ("429", "RESOURCE_EXHAUSTED", "Quota exceeded", "limit: 0")


def _is_retryable(exc: BaseException) -> bool:
    """Quota errors won't recover within seconds — don't retry them."""
    return not any(m in str(exc) for m in _QUOTA_MARKERS)


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
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(img_bytes).decode()}},
                {"text": PROMPT},
            ],
        }],
    )
    return (resp.text or "").strip()


# ── Public classify function ───────────────────────────────────────────────────

def classify(img_bytes: bytes, img_original, state) -> None:
    """Gemini classification worker — run in a daemon thread."""
    try:
        if _circuit_is_open():
            state.set_status(
                "API paused",
                f"Circuit open after repeated failures — "
                f"retrying in up to {settings.cb_recovery_sec:.0f}s",
            )
            logger.warning("Circuit breaker open — skipping Gemini call.")
            return

        state.set_status("Classifying...", "Sending request to Gemini...")
        logger.info("Gemini request → model=%s  bytes=%d", MODEL_NAME, len(img_bytes))

        raw = _call_gemini(img_bytes)   # tenacity handles up to 3 retries
        _record_success()

        logger.info("Gemini raw response: %s", raw)

        data          = _extract_json(raw)
        label         = str(data.get("category",     "Other")).strip().capitalize()
        description   = str(data.get("description",  "N/A")).strip()
        brand_product = str(data.get("brand_product", "Unknown")).strip()

        if label not in VALID_CLASSES:
            label = "Other"

        state.set_status(label, f"{brand_product} | {description}")
        state.add_to_history(label)

        if label != "Empty":
            save_entry(label, img_original, description, brand_product)

    except Exception as e:
        msg = str(e)
        logger.error("Gemini error: %s", msg)
        try:
            with open(ERR_JSON_FILE, "w", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass

        opened    = _record_failure()
        quota_hit = any(m in msg for m in _QUOTA_MARKERS)

        if quota_hit:
            state.set_status(
                "Quota exceeded (429)",
                "Check ai.dev/rate-limit or enable billing.",
            )
        elif opened:
            state.set_status(
                "Circuit open",
                f"API down — paused {settings.cb_recovery_sec:.0f}s before retry.",
            )
        else:
            state.set_status("Error", "See logs/last_api_error_*.json for details.")

    finally:
        state.finish_classify()
