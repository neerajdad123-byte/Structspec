#!/usr/bin/env python3
"""
Generate a token corpus JSON file from a directory of source-code files.

Usage:
    python -m scripts.generate_token_corpus \
        --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --input-dir /path/to/code/ \
        --output tokens.json \
        --glob "*.py"

This produces a JSON file compatible with StructSpec's --token-json argument.
Each file becomes one example entry, tokenized by the specified model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llama_cpp import Llama


def tokenize_file(model_path: str, file_path: Path, n_ctx: int = 2048) -> dict:
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=0,
        n_ctx=n_ctx,
        logits_all=False,
        verbose=False,
    )
    text = file_path.read_text(encoding="utf-8")
    ids = list(
        llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
    )
    tokens = []
    for tid in ids:
        piece = llm.detokenize([tid]).decode("utf-8", errors="ignore")
        tokens.append({"id": tid, "token": piece})
    return {
        "name": str(file_path),
        "tokens": tokens,
        "code": text,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a StructSpec token corpus JSON from source files"
    )
    ap.add_argument("--model", required=True, help="Path to GGUF model used for tokenization")
    ap.add_argument("--input-dir", required=True, help="Directory containing source files")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--glob", default="*.py", help="File glob pattern (default: *.py)")
    ap.add_argument("--n-ctx", type=int, default=2048, help="Model context size")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input-dir is not a directory: {input_dir}", file=sys.stderr)
        raise SystemExit(2)

    files = sorted(input_dir.rglob(args.glob))
    if not files:
        print(f"WARNING: no files matched '{args.glob}' under {input_dir}", file=sys.stderr)

    print(f"Tokenizing {len(files)} files with {args.model} …")
    corpus: dict[str, dict] = {}
    for idx, fp in enumerate(files, 1):
        print(f"  [{idx}/{len(files)}] {fp}")
        corpus[fp.name] = tokenize_file(args.model, fp, n_ctx=args.n_ctx)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    print(f"\nWrote corpus ({len(corpus)} examples) to {out_path}")


if __name__ == "__main__":
    main()
