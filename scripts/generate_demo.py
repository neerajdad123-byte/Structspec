#!/usr/bin/env python3
"""
Demo helper for generating README visuals and GIFs.

Usage:
    python scripts/generate_demo.py

Then use a screen recorder (e.g., Peek, ScreenToGif, or vhs) to capture the terminal.
Recommended settings: 80×24 terminal, 15 fps, 5–10 second capture.
"""

import sys
from pathlib import Path

# Ensure src is on path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structspec_qwen.cli import benchmark, parse_args


def main():
    # Force a short, visually compelling demo run
    sys.argv = [
        "structspec-qwen",
        "--prompts", "5",
        "--tokens", "60",
        "--k", "6",
        "--rich-viz",
        "--syntax-mode", "cluster",
        "--reject-mode", "seq-bonus",
        "--live-mining",
        "--extra-corpus",
    ]
    benchmark(parse_args())


if __name__ == "__main__":
    main()
