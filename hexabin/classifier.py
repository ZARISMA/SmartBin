"""
hexabin/classifier.py — classification worker.

Dispatches on HEXABIN_CLASSIFY_MODE:

* ``local``  (default) — run the configured LLM backend (hexabin.llm) in
  this process and persist the result locally (plus /api/report when
  EDGE_MODE is enabled). This is the original standalone behavior.
* ``server`` — POST the frame to the central server's /api/edge/classify; the
  server runs the LLM, stores the result, and replies with the actuation
  command ("open module N") which is dispatched here. Nothing is persisted
  on the edge in this mode.

The public entry point ``classify(img_bytes, img_original, state)`` keeps its
signature — it is the daemon-thread target used by utils.launch_classify.
"""

from __future__ import annotations

from .actuator import dispatch
from .config import CLASSIFY_MODE, VALID_CLASSES
from .dataset import _environment_data, save_entry
from .edge_client import EdgeServerError, classify_remote
from .llm import CircuitOpenError, build_backend, circuit_is_open, is_quota_error
from .llm import extract_json as _extract_json  # noqa: F401 — legacy import path
from .log_setup import ERR_JSON_FILE, get_logger
from .settings import settings

logger = get_logger()


def classify(img_bytes: bytes, img_original, state) -> None:
    """Classification worker — run in a daemon thread."""
    try:
        if CLASSIFY_MODE == "server":
            _classify_via_server(img_bytes, state)
        else:
            _classify_local(img_bytes, img_original, state)
    finally:
        state.finish_classify()


def _classify_local(img_bytes: bytes, img_original, state) -> None:
    """Run the LLM backend in-process and persist the result locally."""
    try:
        state.set_status("Classifying...", "Running LLM classification...")

        result = build_backend().classify(img_bytes)

        label = result.category
        state.set_status(label, f"{result.brand_product} | {result.description}")
        state.add_to_history(label)

        if label != "Empty":
            save_entry(
                label,
                img_original,
                result.description,
                result.brand_product,
                confidence=result.confidence,
                backend=result.backend,
            )
            dispatch(label)

    except CircuitOpenError as e:
        state.set_status("API paused", str(e))
        logger.warning("Circuit breaker open — skipping LLM call.")

    except Exception as e:
        msg = str(e)
        logger.error("Classification error: %s", msg)
        try:
            with open(ERR_JSON_FILE, "w", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass

        if is_quota_error(msg):
            state.set_status(
                "Quota exceeded (429)",
                "Check ai.dev/rate-limit or enable billing.",
            )
        elif circuit_is_open():
            state.set_status(
                "Circuit open",
                f"API down — paused {settings.cb_recovery_sec:.0f}s before retry.",
            )
        else:
            state.set_status("Error", "See logs/last_api_error_*.json for details.")


def _classify_via_server(img_bytes: bytes, state) -> None:
    """Send the frame to the central server; apply its result + command."""
    try:
        state.set_status("Classifying...", "Waiting for central server...")

        resp = classify_remote(img_bytes, env=_environment_data())

        result = resp.get("result") or {}
        label = str(result.get("category", "Other")).strip().capitalize()
        if label not in VALID_CLASSES:
            label = "Other"
        brand = str(result.get("brand_product", "Unknown")).strip()
        desc = str(result.get("description", "N/A")).strip()

        state.set_status(label, f"{brand} | {desc}")
        state.add_to_history(label)
        state.warnings.clear("SERVER_UNREACHABLE")

        command = resp.get("command") or {}
        if command.get("action") == "open_module":
            dispatch(label, module=command.get("module"))

    except EdgeServerError as e:
        msg = str(e)
        logger.error("Server classification failed: %s", msg)
        state.set_status("Server error", msg[:80])
        state.warnings.add(
            "SERVER_UNREACHABLE",
            f"Central server classify failed: {msg[:120]}",
            severity="error",
        )
    except Exception as e:
        logger.error("Server classification error: %s", e)
        state.set_status("Error", str(e)[:80])
