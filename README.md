# MRound

**Weight-only quantization for language models, built for Apple Silicon.**

MRound compresses transformer checkpoints to 8, 4 and 2 bits by learning each
weight's rounding decision instead of accepting the nearest code. The learning
runs on the Apple GPU through MLX and Metal, and the result is an MLX checkpoint
that `mlx-lm` loads directly.

It is an independent implementation of SignRound v2, written from the published
method. It is not a fork of any existing quantizer and contains no source copied
from one.

> **Alpha.** The method works and is measured against the reference
> implementation on three models: see [RESULTS.md](RESULTS.md), where every
> number cites the run that produced it. What is not built yet is listed under
> [What is not here](#what-is-not-here). Interfaces may change before 1.0.

---

## Why a Mac

Existing implementations of this method target datacenter accelerators. Their
tuning loops assume CUDA, Gaudi or Xeon, and their engineering assumes discrete
device memory: explicit transfers across a bus, activation offload, careful
staging of what fits in 24 or 80 gigabytes.

A Mac is a different machine. Unified memory puts an entire model and its
calibration activations in one address space with no copy across a bus, and a
128 GB Mac Studio has more memory available to a quantization run than most
single discrete GPUs. That inverts the usual constraint: the Mac is a natural
place to *produce* low-bit models, not only to run them.

MRound is built on that premise, and only on that premise. There is no CUDA
path, no ROCm path and no portability layer. Every design decision may assume
unified memory and a Metal GPU.

---

## Install

Requires macOS on Apple Silicon, Python 3.11 or later, and MLX 0.32 or later.
Intel Macs are not supported.

```bash
git clone https://github.com/fbaldassarri/mround.git
cd mround
conda create -n mround python=3.11 && conda activate mround
pip install -e '.[models]'
```

Conda rather than a plain virtual environment for one specific reason: MLX needs
a native arm64 Python and will not install under Rosetta. Check with

```bash
python -c "import platform; print(platform.processor())"   # must print: arm
```

and if it prints `i386` on an M-series Mac, the environment is emulated. The
`models` extra pulls in `mlx-lm` and `transformers`, which is what loads a
checkpoint; the numerical core and the test suite need neither.

Then ask the machine whether it is ready:

```bash
mround doctor
```

---

## Use

Quantize a model to 4 bits and score it:

```bash
mround quantize mlx-community/SmolLM2-135M-Instruct -o ./smollm2-4bit --bits 4
mround eval ./smollm2-4bit
```

Mixed precision, choosing a width per layer from a menu under an exact size
budget:

```bash
mround quantize mlx-community/Qwen2.5-0.5B-Instruct -o ./qwen-mixed \
    --average-bits 2.5 --options 2,3,4
```

The same thing from Python:

```python
from mround import api

result = api.quantize(
    "mlx-community/Qwen2.5-0.5B-Instruct",
    output="./qwen-mixed",
    average_bits=2.5,
)
```

Every run writes its full configuration beside the checkpoint, so any result can
be reproduced from the artifact alone. [GETTING-STARTED.md](GETTING-STARTED.md)
has the longer tour, including the two examples that show what the learning
actually buys over round-to-nearest.

---

## What it scores

The acceptance criterion for this project is not a threshold somebody chose. It
is Intel AutoRound's own result at the same model, the same scheme, the same
calibration data and the same evaluator, with both sides' absolute perplexities
reported so that a weak opponent cannot flatter a broken model.

| scheme | model | reference | MRound | |
|---|---|---|---|---|
| uniform 4 bit | Qwen2.5-0.5B-Instruct | 15.0777 | 15.0807 | +0.02 percent |
| uniform 4 bit | SmolLM2-135M-Instruct | 22.1758 | 18.9400 | **-14.59 percent** |
| uniform 2 bit | Qwen2.5-0.5B-Instruct | 52.7161 | 39.1842 | **-25.67 percent** |
| mixed 2.5 bit | Qwen2.5-0.5B-Instruct | 28.2033 | 22.3707 | **-20.68 percent** |
| mixed 2.5 bit | SmolLM2-135M-Instruct | 52.0300 | 30.3866 | **-41.60 percent** |

Perplexity on wikitext2, lower is better. The three-seed rows are means over
calibration seeds 42, 43 and 44, and in both of those the worst MRound draw
still beats the best reference draw. MRound's checkpoints run 2.4 to 4.9 percent
larger at the same nominal width, which is stated in every table rather than left
out.

[RESULTS.md](RESULTS.md) has all of it: the per seed numbers, the sizes, the
original model's perplexity beside every comparison, what happens when the
reference is run at its own strongest recipe, which release of the reference each
row used, and what these measurements do not show. Every figure there cites a
line in [`benchmarks/ledger.jsonl`](benchmarks/ledger.jsonl), which is append
only, so a correction is a new row and nothing is quietly revised.

The honest caveat, stated once here and again there: these are three models, all
under two billion parameters, scored by perplexity on one corpus. The largest
model measured shows the smallest margin. Nothing here is a claim about seven
billion.

---

## How it works

Round-to-nearest quantization decides each weight independently, which throws
away the one thing that matters: whether rounding *this* weight up and *that*
one down would leave the block's output closer to the original.

SignRound learns that jointly. For each transformer block it introduces a small
perturbation per weight, bounded to plus or minus half a quantization step, plus
a pair of learned clipping coefficients, and optimizes them with signed gradient
descent so the quantized block reproduces the original block's output on
calibration data. The weights themselves never move. Only the rounding does.

SignRound v2 adds three things that matter at low widths: a searched scale
initialization rather than the observed range, which is what makes 2 bits a
model rather than noise; an outlier suppressed loss; and gradient weighted
sensitivity scoring, which is what lets a size budget be spent per layer instead
of uniformly.

MRound implements all of it in two layers. `mround/reference/` is a complete
NumPy implementation of the mathematics that runs on any machine, needs no model
and has exhaustive unit tests with finite difference verified gradients.
`mround/core/` is the MLX port, mirroring it function for function, checked
against it by a parity suite. When the two disagree, the NumPy one is presumed
right. [DOCUMENTATION.md](DOCUMENTATION.md) is the specification: the formulas,
the architecture, the on-disk formats and the hazards.

---

## What is not here

Named rather than approximated. Every unimplemented entry point raises with a
pointer instead of returning something plausible.

GGUF export and Core ML export are not implemented, so the only output format
today is an MLX checkpoint. Zero-shot task evaluation is not implemented;
perplexity is the only quality measure. Vision and multimodal models are not
supported. Activation quantization is out of scope, as is quantization aware
training, as is any non-Apple hardware target.

Memory requirements are not yet characterized. MRound quantizes one block at a
time, so peak usage should be governed by the largest single block plus cached
activations rather than by the whole model, but that has not been measured and
no figure is claimed until it is.

There is no serving runtime and there will not be one. MRound produces
checkpoints; `mlx-lm` runs them.

---

## Relationship to Intel AutoRound

[Intel AutoRound](https://github.com/intel/auto-round) is the reference
implementation of the method MRound implements, by the people who published it.
This project is grateful for it on two counts: the research, and a rigorous
implementation to check numerical correctness against.

MRound is not a fork and shares no source code with it. The two differ in target
hardware, in framework and in scope. Where their on-disk formats overlap, MRound
implements the format from its observable structure so that checkpoints stay
mutually loadable and nobody is locked into either tool. Both projects are
Apache-2.0.

`scripts/measure_overlap.py` is the check that keeps that claim testable rather
than asserted: it compares every function in MRound against every function in an
installed reference, both as identifier blind token streams and as syntax tree
shapes, and prints the closest match found. Run it against any corpus you like.

---

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the setup, the verification gate and the
handful of constraints that a change has to respect. The short version: run

```bash
ruff format . && ruff check --fix .
mypy mround tests scripts examples
pytest tests/unit -q
python scripts/check_module_layout.py
```

before proposing anything. It is what continuous integration runs and it takes
seconds. The unit suite needs neither MLX nor a Mac, on purpose.

---

## License

Apache License 2.0. See [LICENSE.md](LICENSE.md) and [NOTICE](NOTICE) for
attribution, provenance and trademarks.

## Citation

MRound implements methods from:

```bibtex
@inproceedings{cheng2024signround,
  title     = {Optimize Weight Rounding via Signed Gradient Descent
               for the Quantization of LLMs},
  author    = {Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and
               Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi},
  booktitle = {Findings of EMNLP},
  year      = {2024},
  eprint    = {2309.05516}
}

@misc{cheng2025signroundv2,
  title         = {SignRoundV2: Toward Closing the Performance Gap in Extremely
                   Low-Bit Post-Training Quantization for LLMs},
  author        = {Cheng, Wenhua and Zhang, Weiwei and Guo, Heng and
                   Shen, Haihao and Ma, Zaner},
  year          = {2025},
  eprint        = {2512.04746},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
