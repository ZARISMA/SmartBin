"""
mainauto.py — Smart Waste AI (Automatic Gate Mode)

Local presence detection gates all Gemini API calls:
  Calibrating → Ready/IDLE → Detected → Classified → Ready/IDLE

Controls:
  q — quit
  c — force-classify current frame
  r — reset background model from current frame

CLI overrides (all also settable via env vars or .env):
  --model NAME          Gemini model  (SMARTWASTE_MODEL_NAME)
  --threshold FLOAT     Pixel-diff threshold  (SMARTWASTE_MOTION_THRESHOLD)
  --detect-n INT        Consecutive detections to confirm  (SMARTWASTE_DETECT_CONFIRM_N)
  --empty-n INT         Consecutive empties to clear  (SMARTWASTE_EMPTY_CONFIRM_N)
  --location NAME       Deployment location tag  (SMARTWASTE_LOCATION)
"""

from __future__ import annotations

import argparse
import os


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmartWaste AI — Auto-gate mode (dual OAK cameras)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", metavar="NAME", help="Gemini model name")
    p.add_argument(
        "--threshold", type=float, metavar="FLOAT", help="Pixel-diff motion threshold (0–255)"
    )
    p.add_argument(
        "--detect-n",
        type=int,
        metavar="N",
        help="Consecutive detections required to confirm presence",
    )
    p.add_argument(
        "--empty-n", type=int, metavar="N", help="Consecutive empty checks required to clear state"
    )
    p.add_argument("--location", metavar="NAME", help="Deployment location written to dataset")
    return p.parse_args()


def main() -> None:
    args = _parse()

    # Propagate CLI overrides as env vars BEFORE importing smartwaste so that
    # Settings() picks them up when modules are first imported.
    if args.model:
        os.environ["SMARTWASTE_MODEL_NAME"] = args.model
    if args.threshold is not None:
        os.environ["SMARTWASTE_MOTION_THRESHOLD"] = str(args.threshold)
    if args.detect_n is not None:
        os.environ["SMARTWASTE_DETECT_CONFIRM_N"] = str(args.detect_n)
    if args.empty_n is not None:
        os.environ["SMARTWASTE_EMPTY_CONFIRM_N"] = str(args.empty_n)
    if args.location:
        os.environ["SMARTWASTE_LOCATION"] = args.location

    from smartwaste.app import run_loop
    from smartwaste.strategies import PresenceGateStrategy

    run_loop(PresenceGateStrategy())


if __name__ == "__main__":
    main()
