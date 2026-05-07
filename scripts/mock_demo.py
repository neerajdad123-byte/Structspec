#!/usr/bin/env python3
"""
Mock demo for screen-recording the Rich visualization.

Does not require a real model — it replays a realistic token sequence
that looks like speculative decoding of a Python class.

Usage:
    python scripts/mock_demo.py

Then use a screen recorder (ScreenToGif, Peek, vhs) to capture the terminal.
Recommended: 80 cols × 24 rows terminal, 15 fps, 8–12 second capture.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from structspec_qwen.viz import RichVisualizer, TokenEvent


# Realistic speculative-decoding trace for a Python Node class
# Each tuple: (token_text, status, tier, delay_seconds)
DEMO_TRACE: list[tuple[str, str, str, float]] = [
    # class Node:
    ("class", "accepted", "syntax_code_fence_class", 0.06),
    (" N", "accepted", "", 0.05),
    ("ode", "accepted", "", 0.05),
    (":", "accepted", "syntax_def_colon", 0.05),
    ("\n", "bonus", "", 0.12),
    #     def __init__(self):
    ("    ", "accepted", "syntax_indent", 0.05),
    ("def", "accepted", "syntax_code_fence_def", 0.05),
    (" __", "accepted", "syntax_dunder_init", 0.05),
    ("init", "accepted", "syntax_dunder_init", 0.05),
    ("__(", "accepted", "syntax_dunder_paren", 0.05),
    ("self", "accepted", "syntax_dunder_self", 0.05),
    (")", "accepted", "syntax_super_close", 0.05),
    (":", "accepted", "syntax_def_colon", 0.05),
    ("\n", "bonus", "", 0.12),
    #         self.data = data
    ("        ", "accepted", "syntax_indent", 0.05),
    ("self", "accepted", "syntax_dunder_self", 0.05),
    (".", "accepted", "", 0.04),
    ("data", "accepted", "", 0.05),
    (" =", "accepted", "syntax_pluseq_space", 0.04),
    (" ", "accepted", "syntax_pluseq_space", 0.04),
    ("data", "rejected", "", 0.04),
    ("value", "bonus", "", 0.12),
    ("\n", "accepted", "syntax_return_terminal", 0.05),
    #         self.next = None
    ("        ", "accepted", "syntax_indent", 0.05),
    ("self", "accepted", "syntax_dunder_self", 0.05),
    (".", "accepted", "", 0.04),
    ("next", "accepted", "", 0.05),
    (" =", "accepted", "syntax_pluseq_space", 0.04),
    (" ", "accepted", "syntax_pluseq_space", 0.04),
    ("None", "accepted", "syntax_return_terminal", 0.05),
    ("\n", "bonus", "", 0.12),
    #     def append(self, data):
    ("    ", "accepted", "syntax_indent", 0.05),
    ("def", "accepted", "syntax_code_fence_def", 0.05),
    (" ", "accepted", "", 0.04),
    ("append", "accepted", "", 0.05),
    ("(", "accepted", "syntax_range_paren", 0.04),
    ("self", "accepted", "syntax_dunder_self", 0.05),
    (",", "accepted", "", 0.04),
    (" ", "accepted", "", 0.04),
    ("data", "accepted", "", 0.05),
    (")", "accepted", "", 0.04),
    (":", "accepted", "syntax_def_colon", 0.05),
    ("\n", "bonus", "", 0.12),
    #         pass
    ("        ", "accepted", "syntax_indent", 0.05),
    ("pass", "accepted", "syntax_statement_end", 0.05),
    ("\n", "bonus", "", 0.12),
    #     def __str__(self):
    ("    ", "accepted", "syntax_indent", 0.05),
    ("def", "accepted", "syntax_code_fence_def", 0.05),
    (" __", "accepted", "syntax_dunder_init", 0.05),
    ("str", "accepted", "", 0.05),
    ("__(", "accepted", "syntax_dunder_paren", 0.05),
    ("self", "accepted", "syntax_dunder_self", 0.05),
    (")", "rejected", "", 0.04),
    (")", "bonus", "", 0.12),
    (":", "accepted", "syntax_def_colon", 0.05),
    ("\n", "bonus", "", 0.12),
    #         return str(self.data)
    ("        ", "accepted", "syntax_indent", 0.05),
    ("return", "accepted", "syntax_return_terminal", 0.05),
    (" ", "accepted", "", 0.04),
    ("str", "accepted", "", 0.05),
    ("(", "accepted", "syntax_range_paren", 0.04),
    ("self", "accepted", "syntax_dunder_self", 0.05),
    (".", "accepted", "", 0.04),
    ("data", "rejected", "", 0.04),
    ("value", "bonus", "", 0.12),
    (")", "accepted", "", 0.04),
    ("\n", "bonus", "", 0.12),
]


def main() -> None:
    viz = RichVisualizer(max_tokens=80, greedy_reference_speed=10.0)
    current_len = 0
    passes = 1
    accepted = 0
    proposed = 0

    with viz:
        for text, status, tier, delay in DEMO_TRACE:
            time.sleep(delay)
            event = TokenEvent(text, status, tier)
            if status in ("accepted",):
                accepted += 1
                proposed += 1
            elif status == "rejected":
                proposed += 1
            current_len += 1
            passes += 1
            viz.update(
                current_len=current_len,
                passes=passes,
                accepted_draft=accepted,
                proposed=proposed,
                token_events=[event],
            )
        # Stay on screen so the user can admire the final state
        time.sleep(2.0)


if __name__ == "__main__":
    main()
