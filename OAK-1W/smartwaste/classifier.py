import base64
import json
import os

from google import genai

from .config import MODEL_NAME, VALID_CLASSES
from .dataset import save_entry
from .log_setup import ERR_JSON_FILE, get_logger
from .prompt import PROMPT

logger = get_logger()


def _build_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set. Use: $env:GEMINI_API_KEY='YOUR_KEY'")
    return genai.Client(api_key=key)


client = _build_client()


def _extract_json(text: str) -> dict:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        return json.loads(t[start:end + 1])
    raise json.JSONDecodeError("No JSON object found", t, 0)


def classify(img_bytes: bytes, img_original, state) -> None:
    """Gemini classification worker — run in a daemon thread."""
    try:
        state.set_status("Classifying...", "Sending request to Gemini...")
        logger.info("Gemini request -> model=%s, bytes=%d", MODEL_NAME, len(img_bytes))

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

        raw = (resp.text or "").strip()
        logger.info("Gemini raw response: %s", raw)

        data         = _extract_json(raw)
        label        = str(data.get("category",     "Other")).strip().capitalize()
        description  = str(data.get("description",  "N/A")).strip()
        brand_product = str(data.get("brand_product", "Unknown")).strip()

        if label not in VALID_CLASSES:
            label = "Other"

        state.set_status(label, f"{brand_product} | {description}")

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

        quota_hit = any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "Quota exceeded", "limit: 0"))
        if quota_hit:
            state.set_status("Quota exceeded (429)", "Check ai.dev/rate-limit or enable billing.")
        else:
            state.set_status("Error", "See logs/last_api_error_*.json for details.")

    finally:
        state.finish_classify()
