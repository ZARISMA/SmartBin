"""
smartwaste/settings.py — centralised, layered configuration.

Priority (highest → lowest):
  1. CLI args  (entry points set SMARTWASTE_* env vars before importing this module)
  2. Shell environment variables
  3. .env file in the project root
  4. Defaults defined in this file

All overridable settings use the ``SMARTWASTE_`` prefix except ``GEMINI_API_KEY``,
which is accepted both bare and prefixed.

Example .env::

    GEMINI_API_KEY=your_key_here
    SMARTWASTE_MODEL_NAME=gemini-2.0-flash
    SMARTWASTE_MOTION_THRESHOLD=10.0
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env lives next to pyproject.toml, one level above this package
_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SMARTWASTE_",
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini API ─────────────────────────────────────────────────────────────
    # Accept both bare GEMINI_API_KEY (the standard name) and the prefixed variant
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "SMARTWASTE_GEMINI_API_KEY"),
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
    db_name: str = "smartwaste"
    db_user: str = "smartwaste"
    db_password: str = "smartwaste"

    # ── Web UI ─────────────────────────────────────────────────────────────────
    web_host: str = "0.0.0.0"
    web_port: int = 8000


# Module-level singleton — imported by config.py and classifier.py
settings = Settings()
