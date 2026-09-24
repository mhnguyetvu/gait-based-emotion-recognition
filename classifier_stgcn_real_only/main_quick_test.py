"""Backward-compatible entry point for the required tiny-overfit diagnostic."""

import sys

from main import main


if __name__ == "__main__":
    if "--tiny-only" not in sys.argv:
        sys.argv.append("--tiny-only")
    main()
