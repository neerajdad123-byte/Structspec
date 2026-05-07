# Contributing to StructSpec

Thank you for your interest in making StructSpec faster, more accurate, and more useful!

## How to Contribute

### Reporting Issues
- Use [GitHub Issues](https://github.com/neerajanand/strictspec/issues) for bugs and feature requests.
- Include reproduction steps, version, backend, and environment details.
- Attach trace CSVs when debugging speculative-decoding behavior.
- Search existing issues before opening a new one.

### Pull Requests
1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, run the benchmark suite.
3. If you've changed APIs, update the README.
4. Ensure your code follows the existing style (PEP 8, type hints).
5. Fill out the pull request template completely.

### Good First Issues
Look for issues labeled [`good first issue`](https://github.com/neerajanand/strictspec/labels/good%20first%20issue) — these are great entry points for new contributors.

## Development Setup

```bash
git clone https://github.com/neerajanand/strictspec.git
cd strictspec
pip install -e ".[all]"
```

### Running Benchmarks
```bash
python -m benchmarks.bench_suite --model /path/to/model.gguf --token-json /path/to/tokens.json
```

### Running Tests
```bash
pytest tests/
```

## Code Style
- Python 3.10+ with type hints.
- Black-compatible formatting (we may add a formatter in CI).
- Keep CLI flags backward-compatible where possible.
- Add docstrings for public functions.

## Community
- Join [GitHub Discussions](https://github.com/neerajanand/strictspec/discussions) for Q&A and ideas.
- Follow updates on Twitter/X and share your benchmarks!
