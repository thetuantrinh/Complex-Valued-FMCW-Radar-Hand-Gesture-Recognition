# Contributing

Thanks for your interest in this project. It accompanies an IEEE TAES article,
so contributions are welcome with one standing constraint: **changes must not
silently alter the published architecture or its reported results.**

## Development setup

```bash
git clone https://github.com/thetuantrinh/Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition.git
cd Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition

conda create -n cvradar python=3.9 -y
conda activate cvradar
pip install -e ".[dev]"
```

Python 3.8–3.10 only: `keras-complex` requires `tensorflow < 2.16`, which has no
wheels for newer interpreters.

## Before opening a pull request

```bash
make check     # ruff lint + format check, mypy, pytest
```

Or individually:

```bash
ruff check . && ruff format --check .
mypy
pytest
```

## Conventions

- **Style** — ruff enforces formatting and lint (see `pyproject.toml`). Run
  `make format` rather than hand-aligning code.
- **Docstrings** — Google style, on every public function and class. Say what a
  function is for, not just what it does.
- **Typing** — annotate public signatures; `mypy` runs over `src/cvradar`.
- **Configuration** — new tunables belong in `src/cvradar/config.py` as typed
  dataclass fields, never as module-level constants or hardcoded literals.
  Unknown keys in a YAML file are rejected, so add the field before using it.
- **Paths** — never hardcode a dataset or checkpoint location. Read it from
  `DataConfig`, which resolves `$CVRADAR_DATA_ROOT` and CLI overrides.
- **No import-time side effects** — importing a module must not touch the
  filesystem, glob a dataset, or build a model.

## Tests

`pytest` must pass without TensorFlow installed: tests that need TensorFlow,
`tensorflow-probability` or `keras-complex` are gated by the markers in
`tests/conftest.py` and skip cleanly. Keep it that way, so configuration, label
encoding and CLI logic stay testable in a lightweight environment.

When fixing a bug, add the regression test first.

## Changes that affect published results

The released checkpoints in `checkpoints/` depend on two things that must not
drift:

1. **The class index order** in `src/cvradar/data/labels.py`. It reproduces the
   alphabetical ordering the original `LabelEncoder` produced; changing it
   invalidates every released checkpoint.
2. **The layer semantics** in `src/cvradar/models/layers.py`, including the
   stride applied to both factors of the (2+1)D decomposition.

If a change to either is genuinely warranted, call it out explicitly in the pull
request and explain the effect on the reported metrics.

## Questions

Open an issue, or contact the corresponding author,
[Dr. Minhhuy Le](mailto:huy.leminh@phenikaa-uni.edu.vn).
