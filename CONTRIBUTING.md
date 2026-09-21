# Contributing

Contributions are welcome. This document is short because most of what a
contributor needs to know is either in [DOCUMENTATION.md](DOCUMENTATION.md),
which is the specification, or enforced by the test suite.

## Setup

```bash
git clone https://github.com/fbaldassarri/mround.git
cd mround
conda env create -f environment.yml
conda activate mround-dev
pre-commit install
```

The environment name is not load-bearing; every script uses whichever Python is
active. Creating one yourself works equally well as long as you finish the job:

```bash
conda create -n mround-dev python=3.11 && conda activate mround-dev
pip install -e '.[dev]'
```

Stopping after `conda create` is the one trap worth naming. It never runs the
pip section of `environment.yml`, so NumPy, pytest and the package itself are
all missing, and everything afterwards fails for that single reason.

Conda rather than a plain virtual environment because MLX needs a native arm64
Python and cannot install under Rosetta. Conda supplies the interpreter and pip,
and every dependency comes from PyPI: mixing a conda-forge NumPy with a pip MLX
is the standard route to two NumPy builds and an ABI mismatch.

MLX and PyTorch are both optional here. The unit suite runs without either, on
any machine, which is deliberate: it is a standing check that the
framework-independent layers really are independent of the framework. Tests that
need MLX, a model or the reference corpus mark themselves and skip when their
prerequisites are absent, so a fresh clone has a green suite.

## The verification gate

Run this before proposing anything. It is what continuous integration runs and
it takes seconds.

```bash
ruff format .                          # format
ruff check --fix .                     # lint
mypy mround tests scripts examples     # types, strict
pytest tests/unit -q                   # unit suite, no MLX required
python scripts/check_module_layout.py  # tree matches DOCUMENTATION.md section 4
```

The last one is worth understanding rather than merely satisfying. It parses the
module diagram out of DOCUMENTATION.md and compares it against the tree in both
directions: adding a module without documenting it fails, and so does
documenting one that does not exist. When it complains, fix whichever of the two
is actually wrong rather than editing the diagram reflexively.

A change that touches the numerics also needs the MLX-gated tests, which means
an Apple Silicon Mac. Metal is what ships and its kernels measurably differ from
MLX's CPU backend, so a suite that is green on CPU and red on Metal is a real
finding rather than a flake. `pip install 'mlx[cpu]'` on Linux runs the whole
suite and makes a useful pre-flight, but it is not the gate.

## Constraints a change has to respect

These are not style preferences. Each one has a recorded reason and a test.

**Apple Silicon only.** No CUDA path, no ROCm path, no device abstraction layer
whose purpose is portability to other vendors' hardware. If a design feels
awkward because it assumes unified memory and a Metal GPU, that is the design
working correctly.

**MLX is the framework.** Nothing under `mround/` may import PyTorch; a test
enforces it. PyTorch appears only in parity tooling that generates reference
fixtures, and is never a runtime dependency.

**`mround/reference/` never imports MLX.** It is the oracle the MLX
implementation is checked against, not scaffolding to be replaced. A test
enforces that too.

**No copied source.** Reading other implementations to understand behavior is
expected. Transcribing them is not. If you find yourself reproducing a function
structurally line by line, stop. Formats are the one exception, and only in one
direction: matching an on-disk representation so that checkpoints interoperate
is legitimate, because a format is not code. Implement it from its observable
structure.

**Apache-2.0 headers.** Every source file opens with exactly this, before the
module docstring, and a test fails the build when one is missing:

```python
# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
```

**Layering.** The optimizer layer does not know what a transformer is. The
planner layer does not know what MLX is. The pipeline layer holds the
model-specific knowledge. This is what makes components testable without loading
a model, which is what makes iteration fast, and a change that violates it will
be asked to move rather than merged.

## Numerics in particular

Section 5 of DOCUMENTATION.md fixes the scale formulas, parameter ranges,
precision policy and loss definitions. Read it before changing any of them.
Where the specification is silent on something you need, that is a gap: raise it
in an issue and let it be decided explicitly. A plausible default filled in
silently produces models that are subtly worse in ways no test catches, which is
the characteristic failure mode of this kind of code.

Section 7 lists the hazards. Three of them have already cost real time and are
worth reading even if you are only passing through: the rounding tie rule and
the clamp boundary gradient, both of which look like edge cases and are not,
because the value reaching the clamp has already been rounded and lands on the
boundary constantly; the bounded rounding perturbation, whose projection is
load-bearing rather than hygiene, since omitting it gives a plausible
implementation of a different algorithm that passes every test you thought to
write; and the fact that float32 does not mean float32 inside a matrix product,
which makes a small residual formed by subtracting two large products mostly
noise.

## Tests

Anything mathematical gets a unit test against a hand computed result on
synthetic data, with no model loaded. Anything that must match reference
behavior gets a parity test against the fixture corpus. Do not write a test that
loads a real model to verify arithmetic.

Small models first, always. Prove it under a billion parameters before trying it
at seven. A change that only works at a scale you cannot iterate on is not yet
finished.

## Measurements

If a change moves a quality, time or memory number that anybody might quote, it
gets a line in `benchmarks/ledger.jsonl` the day it is produced, written through
`mround/eval/results.py`. Unknown fields stay null rather than guessed, and
corrections are new lines, never edits. `python scripts/capture_environment.py`
records the machine's package inventory and says whether it has drifted since
the last recorded run, which is worth doing before a long measurement rather
than after.

A number in the documentation must be traceable to a run. If you cannot verify a
performance claim, please do not write it down.

## Pull requests

State what changed and why. If the change implements a decision that was
discussed in an issue, reference it. Small and correct beats large and
plausible, particularly here, where a subtle numerical regression can pass a
full test suite and only show up as a model that is slightly worse at something
nobody measured.

By contributing you agree that your contribution is licensed under the Apache
License 2.0, as the rest of the project is.
