"""
smartwaste/control.py — unified runnable entry point.

Reads ``SMARTWASTE_CAMERA_MODE`` (pipeline) and ``SMARTWASTE_STRATEGY``
(manual/auto, only used when pipeline is dual-OAK) and launches the correct
capture loop with a shared ``AppState``.

When the admin triggers a restart via the dashboard (``state.request_restart``),
the process exits with code 0 — Docker's ``restart: unless-stopped`` brings it
back up with the new env vars that the server wrote before issuing the command.
"""

from __future__ import annotations

import argparse
import os
import sys

from .log_setup import get_logger
from .state import AppState

logger = get_logger()

VALID_PIPELINES = ("oak", "oak-native")
VALID_STRATEGIES = ("manual", "auto")


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmartWaste AI — unified edge runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pipeline",
        choices=VALID_PIPELINES,
        help="Camera pipeline (overrides SMARTWASTE_CAMERA_MODE)",
    )
    p.add_argument(
        "--strategy",
        choices=VALID_STRATEGIES,
        help="Strategy for dual-OAK pipeline (overrides SMARTWASTE_STRATEGY)",
    )
    return p.parse_known_args()[0]


def _resolve_pipeline() -> str:
    args = _parse_cli()
    if args.pipeline:
        return str(args.pipeline)
    env = os.environ.get("SMARTWASTE_CAMERA_MODE", "").strip().lower()
    if env in VALID_PIPELINES:
        return env
    # "none" / unknown → default to dual OAK
    return "oak"


def _resolve_strategy() -> str:
    args = _parse_cli()
    if args.strategy:
        return str(args.strategy)
    env = os.environ.get("SMARTWASTE_STRATEGY", "").strip().lower()
    if env in VALID_STRATEGIES:
        return env
    return "manual"


def main() -> None:
    pipeline = _resolve_pipeline()
    state = AppState()
    state.set_pipeline(pipeline)

    logger.info("smartwaste-run: pipeline=%s", pipeline)

    exit_code = 0
    try:
        if pipeline == "oak-native":
            from mainoak import main as oak_main

            oak_main(app_state=state)
        else:
            from .app import run_loop
            from .strategies import build_strategy

            strategy_name = _resolve_strategy()
            state.set_strategy(strategy_name)
            run_loop(build_strategy(strategy_name), state=state)
    except Exception as exc:
        logger.exception("smartwaste-run crashed: %s", exc)
        exit_code = 1

    # Exit 0 so Docker's restart policy brings us back up cleanly with any
    # new env vars the admin set before requesting the restart.
    if state.restart_requested:
        logger.info("Restart requested — exiting 0 for supervisor to respawn.")
        sys.exit(0)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
