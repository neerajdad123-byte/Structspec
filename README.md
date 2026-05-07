# StructSpec

> **Accelerate Qwen code generation up to 3× with zero extra VRAM.**
> Structural speculative decoding using Python syntax, Qwen token clusters, and n-gram rules — no draft model required.

---

## Table of Contents

- [Problem & Solution](#problem--solution)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Generating a Token Corpus](#generating-a-token-corpus)
- [Main Modes](#main-modes)
- [Benchmarks](#benchmarks)
- [Live Terminal Visualization](#live-terminal-visualization)
- [Trace Logging](#trace-logging)

---

## Problem & Solution

### The Problem
Standard speculative decoding accelerates LLM inference by using a second, smaller **draft model** to predict tokens. But that comes with serious drawbacks:

- **Extra VRAM** — loading a second model costs 2–7 GB or more
- **Complexity** — orchestrating two models, KV caches, and synchronization is fragile
- **Setup overhead** — finding and quantizing the right draft model for every target model

### The Solution
**StructSpec** takes the opposite path: it drafts tokens directly from **structural knowledge** — no draft model, no extra VRAM, no synchronization overhead.

| Approach | Extra VRAM | Setup Complexity | Speedup |
|---|---|---|---|
| Standard Speculative Decoding | High (2nd model) | High | 1.5–2.5× |
| **StructSpec** | **Zero** | **Low** | **1.4–3.0×** |

StructSpec mines Qwen token patterns from code corpora, adds Python syntax and indentation-aware chains, and verifies every drafted token with the target model. Accepted tokens reduce target decode passes without loading a separate draft model.

---

## How It Works

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Python Syntax  │────▶│  Draft Proposal  │────▶│ Target Model    │
│  + Qwen Rules   │     │  (k tokens)      │     │  Verification   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │                            │
                              ▼                            ▼
                        ┌──────────┐                ┌──────────┐
                        │ Accepted │                │ Rejected │
                        │ (green)  │                │ (red)    │
                        └──────────┘                └──────────┘
```

1. **Mine** Qwen token n-gram rules from code corpora (offline)
2. **Inject** Python syntax rules (`for` → `in`, `:` → indent, `def __` → `init`, ...)
3. **Propose** a chain of up to *k* draft tokens based on current context
4. **Verify** every drafted token against the target model in a single forward pass
5. **Learn** from live output during a session to improve hit rates
6. **Recover** cheaply from rejections using pending-bonus truncation or sequence replay

The result is correct, greedy-identical output — just faster.

---

## Installation

Install from source:

```bash
git clone https://github.com/neerajdad123-byte/Structspec.git
cd Structspec
pip install -e ".[all]"
```

> **Requirements:** Python 3.10+, a Qwen GGUF model, and the `llama-cpp-python` backend.
>
> **Model Compatibility:** The built-in pattern miner and syntax rules are currently optimized for **Qwen** tokenization and Python code generation. You can run other GGUF models via `--model`, but you will see a warning and performance may vary. See [Generating a Token Corpus](#generating-a-token-corpus) for adapting StructSpec to new models.

---

## Quick Start

```bash
# Basic benchmark (20 prompts, 100 tokens)
structspec-qwen \
  --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --token-json /path/to/engineering_dsa_tokens.json \
  --prompts 20 --tokens 100 --k 6 --live-mining

# Interactive rich terminal visualization
structspec-qwen \
  --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --token-json /path/to/engineering_dsa_tokens.json \
  --rich-viz --prompts 5 --tokens 80

# Observe how Qwen tokenizes your prompt
structspec-qwen \
  --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --token-json /path/to/engineering_dsa_tokens.json \
  --observe-tokens 20 --observe-only
```

> **Note:** `--model` and `--token-json` are **required** arguments. There are no built-in default paths, so the tool works out-of-the-box on any OS. Run `structspec-qwen --help` for the full option list.

---

## Detailed Usage

### Required Arguments

| Argument | Description |
|---|---|
| `--model PATH` | Path to the target model GGUF file. |
| `--token-json PATH` | Path to the token corpus JSON used by the pattern miner. |

### Common Options

| Option | Default | Description |
|---|---|---|
| `--tokens N` | `100` | Max tokens to generate per prompt. |
| `--prompts N` | `20` | Number of built-in DSA prompts to run. |
| `--k N` | `6` | Max draft chain length. |
| `--syntax-mode {off,basic,cluster}` | `cluster` | Enable Python syntax backoff rules. |
| `--reject-mode {truncate,seq-bonus,rebuild}` | `seq-bonus` | How to recover from rejected drafts. |
| `--trace-csv PATH` | `qwen_spec_trace.csv` | Where to write the per-pass CSV trace. |
| `--json-output` | off | Emit a final `STRUCTSPEC_JSON:{...}` summary for programmatic use. |

---

## Generating a Token Corpus

The `--token-json` file is the fuel for the pattern miner. You can generate one from your own codebase using the helper script included in the repo:

```bash
python -m scripts.generate_token_corpus \
  --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --input-dir ./my-python-project/ \
  --output my_corpus_tokens.json \
  --glob "*.py"
```

Then use it with StructSpec:

```bash
structspec-qwen \
  --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --token-json my_corpus_tokens.json \
  --prompts 20 --tokens 100
```

Each source file becomes one example in the corpus. The model tokenizes the text and stores token IDs alongside the raw code. Larger and more diverse corpora generally yield better hit rates.

---

## Main Modes

### Fast Structural Mode
Best for everyday use. Syntax rules chain together and rejected drafts are cheaply recovered with truncation.

```bash
structspec-qwen \
  --model /path/to/model.gguf \
  --token-json /path/to/tokens.json \
  --syntax-mode cluster --reject-mode truncate --live-mining
```

### Safe Rejection Mode
When correctness is paramount. Instead of truncating, cheaply replay the KV cache from the last accepted token.

```bash
structspec-qwen \
  --model /path/to/model.gguf \
  --token-json /path/to/tokens.json \
  --syntax-mode cluster --reject-mode seq-bonus --live-mining
```

### Live Terminal Visualization
Real-time token stream with color-coded acceptance, throughput metrics, and a progress bar. Powered by **Rich**.

```bash
structspec-qwen \
  --model /path/to/model.gguf \
  --token-json /path/to/tokens.json \
  --rich-viz --tokens 80 --prompts 5
```

**What you'll see:**
- **Green** `✓` — accepted draft token
- **Red** `✗` — rejected draft token
- **Cyan** `+` — model bonus token (the next ground-truth token)
- **White** `•` — verified pending token

### Draft-Model Fallback
Use a small draft model **only** when structural rules do not fire. StructSpec always tries rules first.

```bash
structspec-qwen \
  --model /path/to/model.gguf \
  --token-json /path/to/tokens.json \
  --draft-model /path/to/draft.gguf --draft-mistake-limit 1
```

### Offline Pattern Observation
Study Qwen token shapes and mined n-grams without running full inference benchmarks.

```bash
structspec-qwen \
  --model /path/to/model.gguf \
  --token-json /path/to/tokens.json \
  --observe-tokens 20 --observe-only
```

### Expanded Benchmark Suite
Compare configurations side-by-side:

```bash
python -m benchmarks.bench_suite \
    --model /path/to/model.gguf \
    --token-json /path/to/tokens.json \
    --prompts 20 --tokens 100
```

---

## Benchmarks

### Local Test Setup

| Field | Value |
|---|---|
| Model | Qwen2.5-7B-Instruct GGUF Q4_K_M |
| Backend | llama.cpp |
| GPU | RTX 4050 Laptop |
| Workload | 20 DSA/code prompts, 100 tokens each |
| Mode | clustered syntax rules + extra DSA corpus + live mining |

### Results

```text
passes    : greedy=2000  spec=1291  speed=1.549×
wall time : greedy=50.235s spec=34.811s speed=1.443×
decode    : 50.055s -> 34.444s
fire rate : 286/1271 = 22.5%
draft acc : 709/782 = 90.7%
```

Individual prompts (heap / linked-list boilerplate) reached **~3× wall-clock speedup**.

---

## Live Terminal Visualization

When `--rich-viz` is enabled, StructSpec renders an interactive terminal dashboard:

```
┌─ Live Token Stream ───────────────────────────────────────────────┐
│ def ✓ __init__ ✓ ( ✓ self ✓ , ✗ [1.61×]                           │
├───────────────────────────────────────────────────────────────────┤
│ Generating tokens ████████████████░░░░░░░░░░░░░░  55% • 0:00:12   │
├───────────────────┬───────────────────────────────────────────────┤
│ Metrics           │ Recent Events                                 │
│ Generated 42/80   │ ✓ self  syntax_dunder_self                    │
│ Passes 28         │ ✓ (     syntax_dunder_paren                   │
│ Draft Acc 90.7%   │ ✗ ,     syntax_indent                         │
│ Throughput 23.4/s │ + ↵     bonus                                 │
│ Speedup 1.61×     │                                               │
└───────────────────┴───────────────────────────────────────────────┘
```

- **Token Stream** — scrollable history of actual token text
- **Progress Bar** — generation progress with elapsed time
- **Metrics Panel** — tokens per second, speedup factor, draft accuracy
- **Recent Events** — last 6 tokens with rule tier metadata

---

## Trace Logging

Every run produces a detailed CSV trace (default: `qwen_spec_trace.csv`):

| Column | Description |
|---|---|
| `ctx_tail_text` | Recent text context |
| `draft_text` | Proposed token text |
| `model_draft_text` | Token the target model actually wanted |
| `accepted_draft` | How many draft tokens were accepted |
| `why` | `draft_accepted`, `draft_token_mismatch`, or `no_pattern_fired` |
| `tier` | Rule tier (e.g., `syntax_indent`, `det_ctx6`) |

Use these traces to debug why a rule was rejected, build new grammar packs, or visualize acceptance heatmaps.
