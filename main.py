"""UPM Match — entry point.

Run with:
    uv run main.py
    uv run main.py --port 8080
    uv run main.py --debug
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UPM Match recommendation server.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug / auto-reload")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO", colorize=True)

    from app import create_app  # noqa: PLC0415 — deferred so dotenv loads first

    flask_app = create_app()
    logger.info("Starting UPM Match on http://{}:{}", args.host, args.port)
    flask_app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
