# Contributing to Arcus

## Setup

```bash
git clone https://github.com/ShakirFarhan/Arcus.git
cd Arcus
uv sync
```

You'll need an ARC API key to run the tool live (`llm.arc.vt.edu` > User
profile > Settings > Account > API keys). Most of the test suite runs
without one; a handful of live tests in `tests/adapters/test_arc_adapter_live.py`
only run if `ARC_API_KEY` is set in your shell:

```bash
uv run pytest tests/ -q
```

## Before opening a PR

- Add a test for any new business logic (routing, reward calculation,
  cache scoring, quality checks). Bug fixes should include a test that
  fails without the fix.
- Keep PRs focused, one change per PR rather than bundling unrelated
  fixes together.
- Run the full test suite locally before pushing; CI runs it again on
  every PR but catching it early saves a round trip.
- Algorithmic functions (bandit logic, reward math, cache scoring)
  should have a docstring explaining *why* a design choice was made,
  not just what the code does.

## Code style

- No new dependencies without discussing them in the PR description
  first, this project deliberately keeps its dependency footprint
  small.
- Comments should explain non-obvious reasoning (a workaround, a
  constraint, a subtle invariant), not restate what the code already
  says.
- Prefer editing existing files and reusing existing patterns over
  introducing new abstractions.

## Reporting bugs

Open an issue with what you ran, what you expected, and what actually
happened. If it's reproducible with `arcus --random`, mention that too,
it helps narrow down whether it's routing-specific.
