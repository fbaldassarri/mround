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
> implementation on five models: see [RESULTS.md](RESULTS.md), where every
> number cites the run that produced it. What is not built yet is listed under
> [What is not here](#what-is-not-here). Interfaces may change before 1.0.

---

## Why a Mac

Existing implementations of this method target datacenter accelerators. Their
tuning loops are written for CUDA, Gaudi or Xeon, reaching an Apple GPU only
through an override, and their engineering assumes discrete device memory:
explicit transfers across a bus, activation offload, careful staging of what
fits in 24 or 80 gigabytes.

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
mround quantize Qwen/Qwen2.5-0.5B-Instruct -o ./qwen-mixed \
    --average-bits 2.5 --options 2,3,4
```

The same thing from Python:

```python
from mround import api

result = api.quantize(
    "Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="./qwen-mixed",
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
| uniform 4 bit | Qwen2.5-0.5B-Instruct | 15.2197 | 15.0260 | -1.27 percent |
| uniform 4 bit | SmolLM2-135M-Instruct | 22.1758 | 18.9400 | **-14.59 percent** |
| uniform 2 bit | Qwen2.5-0.5B-Instruct | 56.4871 | 39.2499 | **-30.52 percent** |
| uniform 2 bit | Qwen2.5-1.5B-Instruct | 24.6497 | 20.4080 | **-17.21 percent** |
| uniform 2 bit | Qwen2.5-3B-Instruct | 19.8912 | 16.8239 | **-15.42 percent** |
| uniform 2 bit | Qwen2.5-7B-Instruct | 13.3478 | 13.0865 | -1.96 percent |
| uniform 2 bit, the reference's own SignRound v2 tuning on | Qwen2.5-0.5B-Instruct | 43.1769 | 39.2499 | -9.10 percent |
| uniform 2 bit, the reference's own SignRound v2 tuning on | Qwen2.5-1.5B-Instruct | 21.7997 | 20.4080 | **-6.38 percent** |
| uniform 2 bit, the reference's own SignRound v2 tuning on | Qwen2.5-3B-Instruct | 17.4211 | 16.8239 | -3.43 percent |
| uniform 2 bit, the reference's own SignRound v2 tuning on | Qwen2.5-7B-Instruct | 13.3598 | 13.0865 | -2.05 percent |
| mixed 2.5 bit | Qwen2.5-0.5B-Instruct | 28.2033 | 22.3707 | **-20.68 percent** |
| mixed 2.5 bit, the reference's own SignRound v2 tuning on | Qwen2.5-0.5B-Instruct | 25.6777 | 22.3707 | **-12.88 percent** |
| mixed 2.5 bit | SmolLM2-135M-Instruct | 52.0300 | 30.3866 | **-41.60 percent** |
| mixed 2.5 bit | Qwen2.5-3B-Instruct | 11.9737 | 11.1699 | **-6.71 percent** |
| mixed 2.5 bit, the reference's own SignRound v2 tuning on | Qwen2.5-3B-Instruct | 11.3996 | 11.1699 | -2.02 percent |

Perplexity on wikitext2, lower is better, every row against auto-round 0.15.0.
auto-round 0.16.0, a later release, wrote the same reference checkpoints as
0.15.0 at the one operating point it has been run at, under both of the
reference's tunings (RESULTS.md, Which release of the reference), and 0.16.1,
published since, wrote them too: seed 42's two 2 bit arms produced again under
it are the 0.16.0 checkpoints to the bit. The reference arms of the
Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct rows ran on
the Linux machine's integrated GPU, the others on CPUs. The three seed rows, the
eight 2 bit ones and the five mixed ones, are means over calibration seeds 42,
43 and 44, and MRound is ahead on every seed of every one of them. In eight its
worst draw still beats the reference's best; in the other five the ranges
overlap: the 2 bit rows with the reference's own SignRound v2 tuning on at 0.5,
3 and 7 billion parameters, the 7 billion row against its default tuning, and
the mixed Qwen2.5-3B-Instruct row against that tuning. The mixed
Qwen2.5-3B-Instruct rows are MRound's allocator scoring layers as the reference
does, with a gradient at each width's own quantized point, which its default
does from 2.5 billion weights in the scored layers up; below that it takes one
gradient at the widest width, against which the other scored 18.12 and 10.91
percent worse on the two smaller models' mixed rows (RESULTS.md, Mixed
precision). With the widest width's gradients at three billion, its default
until that change, MRound missed the bar there, 3.46 percent behind the
reference's own SignRound v2 on the mean and behind both tunings on two seeds of
three; tuned at the reference's own allocation, read back out of the reference's
checkpoints, it was 0.75 percent ahead of that tuning on the mean, so the gap
was the allocator's gradient and not the tuning. The reference keeps its
SignRound v2 tuning behind an option, `enable_alg_ext`, that is off by default,
and six comparisons have been run both ways so far. At 2 bits the margin is 9.10
percent with the option on against 30.52 with it off on Qwen2.5-0.5B-Instruct,
6.38 against 17.21 at 1.5 billion parameters and 3.43 against 15.42 at 3
billion, so at those sizes most of it sat between the reference's two tunings
and not between the two implementations; at 7 billion the reference's two
tunings are level and MRound is about 2 percent ahead of both. At mixed 2.5 bits
it is 12.88 against 20.68 on Qwen2.5-0.5B-Instruct, so less than half of the
20.68 sat there, and on Qwen2.5-3B-Instruct the reference's SignRound v2 is 4.79
percent better than its default, less than MRound's margin over the default,
6.71 (RESULTS.md says what the pairs leave open; the reference's machine changed
operating system release between the two tunings of the two smallest models, and
arms produced again on the new system are the same checkpoints to the bit, while
the 3 and 7 billion pairs ran on one system). The 4 bit Qwen2.5-0.5B-Instruct
row is a tie: the reference's own productions of it at that seed span 1.17
percent, at the same bytes, which MRound reaches at group size 128, the
reference is 0.50 percent ahead, and on the reference's own twelve task
zero-shot suite the two checkpoints' mean accuracies are 0.16 points apart
(47.42 against 47.58 percent), the reference 0.60 points ahead at the same
bytes. At 2 bits on the same model, on MRound's current checkpoints, that suite
puts MRound 1.70 points ahead on the mean, on every seed, against the
reference's default arms, and calls a tie against the arms with the option on,
MRound 0.13 points ahead on the mean and behind on two seeds of three; its
checkpoints of the earlier learning rate (below) were 2.05 and 0.49 points
ahead, both on every seed. At mixed 2.5 bits it is equivocal against the default
arms, MRound 0.69 points ahead on the mean and behind on one seed of three, and
0.10 ahead against the arms with the option on, behind on one seed of three: on
the tasks the option closes most of the distance at both schemes. MRound's
checkpoints run 2.4 to 6.2 percent larger at the same nominal width and group
size 64, the larger figures on the larger models (1.2 percent at 128), which is
stated in every table rather than left out.

The 2 bit rows are MRound at the reference's learning rate below 4 bits, which
it follows since 0.1.0a6 ([CHANGELOG.md](CHANGELOG.md)), every seed of every
size measured again on one tree. At its earlier rate, twice the reference's at
the standard 200 steps, it was 0.78 percent ahead of the SignRound v2 tuning at
1.5 billion parameters and 1.25 at 3 billion, and 5.87 behind at 7 billion;
RESULTS.md keeps those tables too.

[RESULTS.md](RESULTS.md) has all of it: the per seed numbers, the sizes, the
original model's perplexity beside every comparison, what happens when the
reference is run at its stronger recipe, which release of the reference each
row used, and what these measurements do not show. Every perplexity, accuracy,
size and wall clock there cites a line in
[`benchmarks/ledger.jsonl`](benchmarks/ledger.jsonl), which is append only, so a
correction is a new row and nothing is quietly revised; the few supporting
figures that come from a run's own output rather than the ledger (checkpoint
distances, the evaluator check, a peak read from a log) say so where they
appear.

The honest caveat, stated once here and again there: these are five models, of
seven billion parameters at most, scored by perplexity on one corpus, and only
seven comparisons so far by a zero-shot task suite as well, two of them on
MRound's current 2 bit checkpoints, where against the reference's own SignRound
v2 tuning the suite calls a tie. At 2 bits the margin against the reference's
default tuning is 30.52, 17.21, 15.42 and 1.96 percent from 0.5 to 7 billion
parameters, and against its own SignRound v2 tuning 9.10, 6.38, 3.43 and 2.05:
against the method MRound implements the lead shrinks as the model grows, and at
7 billion it is a level result that MRound wins on every seed rather than a
margin to quote alone. The 4 bit rows stop at half a billion parameters, with
the 1.5, 3 and 7 billion parameter models set at 4 bits against round-to-nearest
alone, which MRound beats by 5.65, 7.17 and 0.75 percent and which states no
bar; 8 and 3 bits have one row each, at half a billion parameters against
round-to-nearest alone, level at 8 bits and 56.33 percent ahead at 3; and the
mixed rows stop at three billion, and the size at which MRound's allocator
changes its scoring gradients, 2.5 billion weights in the scored layers, lies
between the two models measured on either side of it, with nothing measured
between them. Outside the rows that say otherwise, every reference arm ran
AutoRound's default tuning: the 2 bit arms at every size and the mixed 2.5 bit
Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct arms have been run with
`enable_alg_ext` as well. The other margins are against the reference as it
ships, and none is against the configuration its documentation recommends at 2
bits, that option together with its stronger recipe.

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
initialization rather than the observed range, which in MRound's own
measurements is what makes 2 bits a model rather than noise; an outlier
suppressed loss; and gradient weighted
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

Named rather than approximated. Every unimplemented entry point raises instead
of returning something plausible.

GGUF export and Core ML export are not implemented, so the only output format
today is an MLX checkpoint. Zero-shot task evaluation runs
(`mround eval --zeroshot`), and seven comparisons have been through it so far,
the 4 bit one, the 2 bit and mixed 2.5 bit ones against each of the reference's
two tunings, and the 2 bit pair again on MRound's current checkpoints, all on
Qwen2.5-0.5B-Instruct; every other number above rests on perplexity alone.
Vision and multimodal models are not supported. Activation quantization is out
of scope, as is quantization aware training, as is any non-Apple hardware
target.

Memory and time are characterized on one machine, a 32 GB Mac. MRound quantizes
one block at a time, and it now reads one block's weights at a time as well,
releasing each block once it is quantized, so a uniform run no longer has to
hold the whole model. A mixed precision run's scoring pass still reads every
weight, but holds one quantized copy of the block weights rather than one per
candidate width, and a tuning step can run its batch in pieces
(`--micro-batches`) for a model whose step does not fit whole (DOCUMENTATION.md
section 2). Each checkpoint's `mround_run.json` records the peak of every stage
of every block, and every run predicts its own peak before it starts, from the
model's configuration alone, by a rule fitted on those records and on a probe of
synthetic blocks on Metal (`scripts/probe_memory.py`): a run predicted past 85
percent of the Mac's memory is refused before the model loads, with the number
of pieces that would fit, and `mround quantize --dry-run` asks without running.
The MLX allocator peak was recorded at two sizes before that change, 13.58 GB on
Qwen2.5-0.5B-Instruct at 4 bits and 27.27 GB on Qwen2.5-1.5B-Instruct at 2 bits,
both at the standard recipe; after it, the second configuration wrote the same
checkpoint to the byte at 19.34 GB ([RESULTS.md](RESULTS.md), row 121 of the
ledger), and Qwen2.5-3B-Instruct at 2 bits peaked at 20.07 GB on each of three
seeds (rows 124 to 126). As of 2026-11-13 every size has been measured on the 32
GB M5 MacBook Pro that every MRound run here was made on, the time being that
machine's; the 2 bit times are the runs behind RESULTS.md's 2 bit tables:

| Model, standard recipe | Peak memory | 4 bits | 2 bits | 4 bit ledger rows |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 10.93 GB | 1h00m to 1h14m | 1h05m to 1h08m | 95 and 233 |
| Qwen2.5-1.5B-Instruct | 14.14 to 14.25 GB | 2h54m | 2h37m to 2h38m | 235 |
| Qwen2.5-3B-Instruct | 20.07 GB | 6h01m | 4h35m to 5h06m | 236 |
| Qwen2.5-7B-Instruct, every step in four pieces | 28.84 GB | 15h50m | 16h29m to 17h39m | 232 |

A mixed 2.5 bit run adds its scoring pass, which at 3 billion parameters peaked
at 23.95 GB and took about an hour. Every run predicts its time before it starts
too, at the pace of that machine undisturbed, by a rule fitted on fifty five of
its runs, which took 0.91 to 1.35 times the rule: the same checkpoint, written
twice to the byte, took 1h00m in September and 1h14m in November. On another Mac
the figure is that machine's, a reference point. Six runs since 2026-11-08 held
one or two blocks 0.32 to 0.83 GB above the line the others follow, which the
memory rule does not predict and the budget's margin covers.

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
