# Demo Assets

## Creating the Hero GIF / Video

### Option 1: Mock Demo (No model required)
The fastest way to generate a flawless visual is using the mock replay script:

```bash
python scripts/mock_demo.py
```

Then use a screen recorder while it runs:
- **Windows**: [ScreenToGif](https://www.screentogif.com/) — select the terminal window, 15 fps, 5–8 s capture
- **Linux**: [Peek](https://github.com/phw/peek) or [vhs](https://github.com/charmbracelet/vhs)
- **macOS**: [vhs](https://github.com/charmbracelet/vhs)

Recommended terminal size: **80 columns × 24 rows** for crisp readability.

Save the recording as `docs/assets/demo.gif` and reference it in the main README.

### Option 2: Real Inference Demo
If you have a Qwen GGUF model locally:

```bash
python scripts/generate_demo.py
```

This runs a short benchmark with `--rich-viz` enabled. Record the same way as above.

## Static Charts

Run the benchmark suite to generate comparison charts:

```bash
python -m benchmarks.bench_suite \
    --model /path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --token-json /path/to/engineering_dsa_tokens.json \
    --prompts 20 --tokens 100
```

Outputs are saved to `benchmarks/reports/`:
- `results.json` — raw data
- `speedup_comparison.png` — grouped bar chart
- `wall_time_comparison.png` — side-by-side wall time

Check these charts into `docs/assets/` and reference them from the README for credibility.
