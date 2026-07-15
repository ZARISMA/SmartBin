"""
main.py — Smart Waste AI (Manual Mode)

Controls:
  c — classify current frame
  a — toggle auto-classify (every --auto-interval seconds)
  q — quit

CLI overrides (all also settable via env vars or .env):
  --model NAME          Gemini model  (HEXABIN_MODEL_NAME)
  --auto-interval SEC   Auto-classify interval  (HEXABIN_AUTO_INTERVAL)
  --location NAME       Deployment location tag  (HEXABIN_LOCATION)
"""

from __future__ import annotations

import argparse
import os


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HexaBin AI — Manual mode (dual OAK cameras)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", metavar="NAME", help="Gemini model name")
    p.add_argument(
        "--auto-interval", type=int, metavar="SEC", help="Seconds between auto-classifications"
    )
    p.add_argument("--location", metavar="NAME", help="Deployment location written to dataset")
    return p.parse_args()


def main() -> None:
    args = _parse()

    # Propagate CLI overrides as env vars BEFORE importing hexabin so that
    # Settings() picks them up when modules are first imported.
    if args.model:
        os.environ["HEXABIN_MODEL_NAME"] = args.model
    if args.auto_interval is not None:
        os.environ["HEXABIN_AUTO_INTERVAL"] = str(args.auto_interval)
    if args.location:
        os.environ["HEXABIN_LOCATION"] = args.location

    from hexabin.app import run_loop
    from hexabin.strategies import ManualStrategy

    run_loop(ManualStrategy())


if __name__ == "__main__":
    main()
