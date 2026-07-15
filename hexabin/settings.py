"""
hexabin/settings.py — centralised, layered configuration.

Priority (highest → lowest):
  1. CLI args  (entry points set HEXABIN_* env vars before importing this module)
  2. Shell environment variables
  3. .env file in the project root
  4. Defaults defined in this file

All overridable settings use the ``HEXABIN_`` prefix except ``GEMINI_API_KEY``,
which is accepted both bare and prefixed.

Example .env::

    GEMINI_API_KEY=your_key_here
    HEXABIN_MODEL_NAME=gemini-2.0-flash
    HEXABIN_MOTION_THRESHOLD=10.0
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives next to pyproject.toml, one level above this package
_ENV_FILE = Path(__file__).parent.parent / ".env"


def _migrate_legacy_env_prefix() -> None:
    """Back-compat: the env prefix was renamed ``SMARTWASTE_`` → ``HEXABIN_``.

    Copy any still-set ``SMARTWASTE_*`` process env var onto its ``HEXABIN_*``
    equivalent when the new name is unset, so pre-rename ``.env`` files and
    already-running containers keep working until they are redeployed.
    """
    for key, value in list(os.environ.items()):
        if key.startswith("SMARTWASTE_"):
            new_key = "HEXABIN_" + key[len("SMARTWASTE_") :]
            os.environ.setdefault(new_key, value)


_migrate_legacy_env_prefix()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HEXABIN_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini API ─────────────────────────────────────────────────────────────
    # Accept both bare GEMINI_API_KEY (the standard name) and the prefixed variant
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "HEXABIN_GEMINI_API_KEY"),
    )
    model_name: str = "gemini-3-flash-preview"

    # ── Deployment ─────────────────────────────────────────────────────────────
    location: str = "Yerevan"

    # ── Camera / capture ───────────────────────────────────────────────────────
    jpeg_quality: int = 85
    crop_percent: float = 0.20
    max_dt: float = 0.25  # max seconds between synced camera frames
    auto_interval: int = 6  # seconds between auto-classifications (manual mode)

    # ── Auto-gate mode (mainauto.py) ───────────────────────────────────────────
    motion_threshold: float = 12.0
    detect_confirm_n: int = 3
    empty_confirm_n: int = 6
    bg_learning_rate: float = 0.03
    bg_warmup_frames: int = 40
    check_interval: float = 0.5

    # ── OAK-D Native mode (mainoak.py) ─────────────────────────────────────────
    oak_display_w: int = 960
    oak_display_h: int = 540
    depth_roi_fraction: float = 0.4
    depth_change_threshold: int = 150
    oak_calib_frames: int = 30
    imu_sample_rate_hz: int = 100
    imu_shock_threshold: float = 1.3
    imu_baseline_samples: int = 50
    drop_flag_duration: float = 3.0
    nn_model_name: str = "mobilenet-ssd"
    nn_shaves: int = 6
    nn_confidence: float = 0.5
    oak_votes_needed: int = 2
    oak_detect_confirm_n: int = 3
    oak_empty_confirm_n: int = 5
    oak_check_interval: float = 0.4
    motion_spike_factor: float = 3.0  # presence score jump multiplier for "drop" event

    # ── API retry / circuit breaker ────────────────────────────────────────────
    api_retry_attempts: int = 3  # max tenacity attempts per classify call
    api_retry_min_wait: float = 1.0  # first backoff wait (seconds)
    api_retry_max_wait: float = 8.0  # max backoff wait (seconds)
    cb_failure_threshold: int = 5  # consecutive failures → open circuit
    cb_recovery_sec: float = 60.0  # seconds to keep circuit open

    # ── Database ───────────────────────────────────────────────────────────────
    db_backend: str = "sqlite"  # "sqlite" or "postgresql"
    db_host: str = "localhost"
    db_port: int = 5432
    # NB: these stay "smartwaste" — the Postgres volume/role was initialized with
    # that name; renaming here would break auth against existing data (see compose).
    db_name: str = "smartwaste"
    db_user: str = "smartwaste"
    db_password: str = "smartwaste"

    # ── Web UI ─────────────────────────────────────────────────────────────────
    web_host: str = "0.0.0.0"
    web_port: int = 8000
    camera_mode: str = "oak"  # "oak", "raspberry", "oak-native", or "none"

    # ── Authentication ─────────────────────────────────────────────────────────
    admin_username: str = "admin"
    admin_password: str = "password123"
    secret_key: str = "hexabin-session-secret-change-in-prod"

    # ── Bin identity ───────────────────────────────────────────────────────────
    bin_id: str = "bin-01"

    # ── Edge mode ──────────────────────────────────────────────────────────────
    edge_mode: bool = False
    server_url: str = ""  # e.g. "http://192.168.1.100:8000"
    edge_api_key: str = ""  # shared secret for authenticating to server
    heartbeat_interval: int = 30  # seconds between heartbeats

    # ── LLM backend (whoever runs the classification) ──────────────────────────
    llm_backend: str = "gemini"  # "gemini", "lmstudio", or "cascade"
    lmstudio_url: str = "http://localhost:1234/v1"  # OpenAI-compatible base URL
    lmstudio_model: str = "google/gemma-4-12b-qat"  # id listed by GET {url}/models
    lmstudio_timeout: float = 120.0  # seconds per local inference call
    # Reasoning models (e.g. gemma-4) burn completion tokens on reasoning BEFORE
    # emitting the JSON — a tight budget yields an empty response.
    lmstudio_max_tokens: int = 2000
    confidence_threshold: float = 0.70  # cascade: escalate to Gemini below this
    llm_max_concurrency: int = 2  # concurrent LLM calls served by the web app
    llm_queue_timeout: float = 30.0  # seconds to wait for a free LLM slot

    # ── Edge classification ────────────────────────────────────────────────────
    classify_mode: str = "local"  # "local" (classify on-device) or "server"
    classify_timeout: float = 180.0  # edge → server classify HTTP timeout (s)

    # ── Actuation ──────────────────────────────────────────────────────────────
    module_map: str = ""  # JSON {"Plastic": 1, ...}; empty → built-in default
    actuator: str = "log"  # "log" or "none" ("gpio" reserved for hardware)

    # ── Ingest limits ──────────────────────────────────────────────────────────
    max_upload_mb: int = 10  # max decoded image size accepted by the server


# Module-level singleton — imported by config.py and classifier.py
settings = Settings()
