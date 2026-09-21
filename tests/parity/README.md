# Parity tests

These compare MRound against the reference implementation at the granularity of
a **single layer**, under **byte-identical inputs** read from a dumped corpus.

Parity is a debugging instrument, not a goal. Bit-exact agreement across
frameworks is unattainable in principle: reduction orders differ, reduced
precision rounds differently, and the pseudo-random streams are unrelated. What
these tests catch is real implementation error, which is what the reference is
actually useful for. See MEMORY.md D-005.

**Seed-matched trajectory comparison is explicitly not attempted.** It does not
work. Dump the inputs and compare outputs.

## Status: planned, not built

This directory holds this README and an empty `__init__.py`. The corpus
generator named below and the tests that would read it have not been written;
`tests/conftest.py`'s `parity` marker exists for them and is unused. What the
project has instead is a model level comparison harness, which quantizes one
model with both implementations under matched settings and scores both with one
evaluator (MEMORY.md D-037), and the function level parity suite in
`tests/unit/test_mlx_parity.py`, which checks the MLX core against the NumPy
reference on synthetic layers.

## Generating the corpus (once the generator exists)

The corpus will not be committed; it is large and it is reproducible. The
intended command, with the reference stack, which runs on Apple Silicon CPU:

```
pip install -e '.[parity]'
python -m tests.parity.generate_corpus --model <small-model> --out tests/fixtures/parity_corpus
```

Tests in this directory will skip themselves when the corpus is absent, so a
fresh clone keeps a green suite.

## What will be compared

- Per-group scales, to within float16 representation error
- Integer codes, by the fraction that differ: identical except where a value
  sits exactly on a rounding boundary (ROADMAP.md, Phase 1b), a small fraction
  at the block level (Phase 2)
- Layer output mean squared error, within a few percent in either direction
