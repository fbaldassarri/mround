# Changelog

Notable changes to MRound, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions follow
[semantic versioning](https://semver.org/spec/v2.0.0.html), with the alpha
caveat that public interfaces may change between pre-release versions without a
major bump. Each version is dated by its release tag.

## 0.1.0a1

The first public release. Everything below is new, so this entry describes the
state of the project rather than a set of changes against a predecessor.

### The method

Learned rounding by signed gradient descent, implemented twice and checked
against itself. `mround.reference` is a complete NumPy implementation of
SignRound v2's mathematics that runs on any machine and needs no model:
quantizers at every supported width, the learned rounding perturbation, both
loss functions, bit packing, and the mixed-precision allocator, with
finite difference verified gradients. `mround.core` is the MLX port, mirroring
it function for function, with a parity suite that agrees at a median of 0.0
percent on learned rounding.

Uniform quantization at 8, 4 and 2 bits, symmetric and asymmetric, group-wise
and per-channel. The searched scale initialization is the default below 4 bits
for symmetric schemes, which is what makes 2 bits produce a model rather than
noise; at 4 bits it is quality neutral by measurement, so the observed range
stays the default there.

Mixed precision over a menu of widths under an exact size budget, with
gradient weighted sensitivity scoring deciding which layers get which width.
Three levers are set by measurement rather than by taste: gradients are taken at
the widest candidate width, tensors outside the block stack are protected at a
higher width, and the candidate menu is graded rather than binary.

### Interfaces

`mround.api.quantize` and `mround.api.quantize_round_to_nearest`, both writing
MLX checkpoints that `mlx-lm` loads, each with a configuration file beside the
checkpoint that reproduces the run.

A command line with four commands: `quantize`, `eval`, `doctor` and `version`.
`quantize` refuses rather than proceeding when it recognizes too few of a
model's blocks, because a run that silently quantizes a third of a model and
reports success is worse than one that stops.

### Measurement

`benchmarks/ledger.jsonl`, append only, one line per quotable number, each
citing a hashed manifest of the machine's full package inventory under
`benchmarks/environments/`. [RESULTS.md](RESULTS.md) reads from it and cites it
by row.

`scripts/measure_overlap.py`, which compares every function in MRound against
every function in an installed reference implementation, as identifier blind
token streams and as syntax tree shapes, so that the claim of independent
implementation can be checked rather than asserted.

### Not implemented

GGUF export, Core ML export and zero-shot task evaluation all raise
`NotImplementedError` with a pointer, and say so in their own docstrings. A test
checks that those markers and the code agree, so neither can drift.

### Known limitations

Measured on three models, all under two billion parameters, on one evaluation
corpus, by perplexity alone. Memory requirements are not yet characterized.
Wall clock comparisons against the reference implementation compare an Apple GPU
against a Linux CPU, because the reference has no Metal path, and are reported
for what a user experiences rather than as a like for like measurement.
