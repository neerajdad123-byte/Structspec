#!/usr/bin/env python3
"""
Expanded benchmark suite for StructSpec.

Runs comparisons across multiple configurations and generates
visual reports (JSON + charts).

Usage:
    python -m benchmarks.bench_suite \
        --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --token-json /path/to/engineering_dsa_tokens.json \
        --prompts 20 --tokens 100
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# Configurations to benchmark
CONFIGURATIONS = [
    {
        "name": "Greedy Baseline",
        "args": ["--no-syntax-patterns", "--k", "0"],
    },
    {
        "name": "Syntax Basic",
        "args": ["--syntax-mode", "basic", "--k", "4"],
    },
    {
        "name": "Syntax Cluster",
        "args": ["--syntax-mode", "cluster", "--k", "6"],
    },
    {
        "name": "Syntax + Live Mining",
        "args": ["--syntax-mode", "cluster", "--k", "6", "--live-mining"],
    },
    {
        "name": "Unsafe Fast (Truncate)",
        "args": [
            "--syntax-mode",
            "cluster",
            "--reject-mode",
            "truncate",
            "--k",
            "6",
        ],
    },
    {
        "name": "Syntax + Extra Corpus",
        "args": [
            "--syntax-mode",
            "cluster",
            "--k",
            "6",
            "--extra-corpus",
            "--live-mining",
        ],
    },
]


def run_one(
    model: str,
    token_json: str,
    prompts: int,
    tokens: int,
    extra_args: list[str],
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "structspec_qwen",
        "--model",
        model,
        "--token-json",
        token_json,
        "--prompts",
        str(prompts),
        "--tokens",
        str(tokens),
    ] + extra_args
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    data: dict[str, Any] = {
        "pass_g": None,
        "pass_s": None,
        "time_g": None,
        "time_s": None,
        "eval_g": None,
        "eval_s": None,
        "pass_speedup": 1.0,
        "time_speedup": 1.0,
        "elapsed_wall_s": dt,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }
    for line in result.stdout.splitlines():
        line = line.strip()
        if "passes" in line and "greedy=" in line and "spec=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("greedy="):
                    data["pass_g"] = int(p.split("=")[1])
                elif p.startswith("spec="):
                    data["pass_s"] = int(p.split("=")[1].rstrip(","))
                elif p.startswith("speed="):
                    try:
                        data["pass_speedup"] = float(p.split("=")[1].rstrip("x"))
                    except ValueError:
                        pass
        elif "wall time" in line and "greedy=" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("greedy="):
                    data["time_g"] = float(p.split("=")[1].rstrip("s,"))
                elif p.startswith("spec="):
                    data["time_s"] = float(p.split("=")[1].rstrip("s,"))
                elif p.startswith("speed="):
                    try:
                        data["time_speedup"] = float(p.split("=")[1].rstrip("x"))
                    except ValueError:
                        pass
        elif "target eval tokens" in line:
            parts = line.split()
            for p in parts:
                if p.startswith("greedy="):
                    data["eval_g"] = int(p.split("=")[1])
                elif p.startswith("spec="):
                    data["eval_s"] = int(p.split("=")[1].rstrip(","))
    return data


def generate_charts(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    labels = [r["config"] for r in results]
    pass_speedups = [r.get("pass_speedup", 1.0) for r in results]
    time_speedups = [r.get("time_speedup", 1.0) for r in results]
    eval_ratios = [
        (r.get("eval_g", 1) / max(1, r.get("eval_s", 1))) if r.get("eval_g") else 1.0
        for r in results
    ]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, pass_speedups, width, label="Pass Reduction", color="#2ecc71")
    bars2 = ax.bar(x, time_speedups, width, label="Wall-Clock Speedup", color="#3498db")
    bars3 = ax.bar(x + width, eval_ratios, width, label="Eval Token Reduction", color="#9b59b6")

    ax.set_ylabel("Speedup / Reduction Factor")
    ax.set_title("StructSpec Benchmark Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="Baseline (1×)")
    ax.legend()
    ax.set_ylim(bottom=0)

    def _annotate(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}×",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    _annotate(bars1)
    _annotate(bars2)
    _annotate(bars3)

    plt.tight_layout()
    plt.savefig(out_dir / "speedup_comparison.png", dpi=150)
    plt.close()

    # Second chart: time breakdown if available
    fig, ax = plt.subplots(figsize=(10, 5))
    configs = [r["config"] for r in results if r.get("time_g") and r.get("time_s")]
    greedy_times = [r["time_g"] for r in results if r.get("time_g") and r.get("time_s")]
    spec_times = [r["time_s"] for r in results if r.get("time_g") and r.get("time_s")]

    x2 = np.arange(len(configs))
    width2 = 0.35
    ax.bar(x2 - width2 / 2, greedy_times, width2, label="Greedy", color="#e74c3c")
    ax.bar(x2 + width2 / 2, spec_times, width2, label="Speculative", color="#2ecc71")
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title("Wall Time Comparison")
    ax.set_xticks(x2)
    ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wall_time_comparison.png", dpi=150)
    plt.close()

    print(f"Saved reports to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="StructSpec Expanded Benchmark Suite")
    ap.add_argument("--model", required=True, help="Path to target model GGUF")
    ap.add_argument("--token-json", required=True, help="Path to token corpus JSON")
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--out-dir", default="benchmarks/reports")
    args = ap.parse_args()

    results: list[dict] = []
    for cfg in CONFIGURATIONS:
        print(f"\n▶ Running: {cfg['name']} …")
        data = run_one(
            args.model,
            args.token_json,
            args.prompts,
            args.tokens,
            cfg["args"],
        )
        print(
            f"  pass_up={data.get('pass_speedup'):.2f}×  "
            f"time_up={data.get('time_speedup'):.2f}×  "
            f"wall={data.get('elapsed_wall_s'):.1f}s  "
            f"rc={data['returncode']}"
        )
        results.append({"config": cfg["name"], "args": cfg["args"], **data})

    generate_charts(results, Path(args.out_dir))


if __name__ == "__main__":
    main()
