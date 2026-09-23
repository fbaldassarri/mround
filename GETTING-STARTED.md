# Getting Started

The practical companion to [README.md](README.md). The README says what MRound
is and why it exists; this says what to type.

---

## What you can and cannot do today

The honest answer first, because it saves an hour of looking for something that
is not built.

**You can quantize a real model on an Apple Silicon Mac,** to 8, 4 or 2 bits,
uniform or mixed precision, from the command line or from Python. The output is
an MLX checkpoint that `mlx-lm` loads directly.

**You can score one,** by perplexity on wikitext2.

**You can run the quantization mathematics on any machine, with no model and no
MLX.** `mround.reference` is a complete NumPy implementation of SignRound v2:
the quantizers, the learned rounding, signed gradient descent, both losses, bit
packing at every supported width, and the mixed-precision allocator. It is the
oracle the MLX implementation is checked against, and it is the fastest way to
understand what the method does.

**You cannot export GGUF or Core ML, and there is no zero-shot task
evaluation.** Each raises with a pointer rather than approximating.

---

## Setup

Requires macOS on Apple Silicon for anything that touches a model. Intel Macs
are not supported.

```bash
git clone https://github.com/fbaldassarri/mround.git
cd mround
conda env create -f environment.yml
conda activate mround-dev
```

The environment name does not matter; scripts use whichever Python is active.
Creating one yourself works equally well as long as you finish the job:

```bash
conda create -n mround-dev python=3.11 && conda activate mround-dev
pip install -e '.[models]'
```

Stopping after `conda create` is the one trap worth naming: it never runs the
pip section of `environment.yml`, so NumPy, MLX and the package itself are all
absent, and everything afterwards fails for that single reason.

Conda rather than a virtual environment for a specific technical reason. MLX
requires a native arm64 Python and cannot install under Rosetta, and conda is
the reliable way to get one; MLX's own installation guide recommends it. Check
that you got a native interpreter:

```bash
python -c "import platform; print(platform.processor())"
```

That must print `arm`. If it prints `i386` on an M-series Mac the environment is
emulated; recreate it with
`CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml`. The test suite
checks this on macOS so the problem surfaces now rather than as a confusing MLX
failure later.

Conda supplies only the interpreter and pip; everything else comes from PyPI.
That is deliberate: mixing a conda-forge NumPy with a pip installed MLX is the
standard route to two NumPy builds and an ABI mismatch that shows up as an
unrelated looking crash.

**Note on platforms.** Nothing in the section on the reference implementation
below requires a Mac. The NumPy layer, the allocator, the tests and the tooling
all run on Linux, which is how continuous integration exercises them. Only
quantizing an actual model needs Apple Silicon.

Then ask the machine whether it is ready:

```bash
mround doctor          # what is installed, what is missing, and why it matters
mround doctor --json   # the same thing for a script; exit 0 means ready
```

---

## Your first quantization, in ten seconds

```bash
python examples/first_quantization.py
```

```
layer: 32 x 128, 256 samples

 bits  group   round-to-nearest        learned   better by
------------------------------------------------------------
    2    128         5.7794e-02     3.3568e-02      41.0%
    3    128         1.3174e-02     1.0097e-02      22.2%
    4    128         3.1418e-03     2.6007e-03      17.2%
    8    128         1.1899e-05     1.1755e-05       1.2%
```

Every row uses the same weights and the same bit budget. The only difference is
whether the rounding decisions were chosen or learned.

That table is the whole argument for the project in one screen.
Round-to-nearest picks the closest representable value for each weight
independently, which is optimal per weight and not optimal for the layer,
because what matters is the error in the layer's output and errors in different
weights interact. Learned rounding optimizes against the output instead. Notice
that the advantage grows sharply as bits get scarcer: negligible at 8 bits,
decisive at 2. Two bits is where the method earns its keep, and 2 bits is what
MRound exists to serve.

---

## Your first real model

```bash
mround quantize mlx-community/SmolLM2-135M-Instruct -o ./smollm2-4bit --bits 4
mround eval ./smollm2-4bit
```

`quantize` prints what it is about to do before it does it: the widths, the
recipe, the calibration corpus and how many of the model's blocks it recognized.
That last number matters, and the command refuses rather than proceeding when it
is too low. A run that silently quantizes a third of a model and reports success
is worse than one that stops.

Mixed precision chooses a width per layer from a menu, under an exact size
budget:

```bash
mround quantize Qwen/Qwen2.5-0.5B-Instruct -o ./qwen-mixed \
    --average-bits 2.5 --options 2,3,4
```

Below 4 bits the searched scale initialization becomes the default for
symmetric schemes, and it is not a detail: on the observed grid, 2 bits never
produced anything better than noise. At 4 bits the search is quality neutral by
measurement, so the observed range stays the default there.

The same from Python, with the configuration written beside the checkpoint so
that any result reproduces from the artifact alone:

```python
from mround import api

result = api.quantize(
    "Qwen/Qwen2.5-0.5B-Instruct",
    output="./qwen-mixed",
    average_bits=2.5,
)
```

Two example scripts go further than the command line does.
`examples/round_to_nearest_model.py` quantizes a model the naive way in seconds,
and `examples/tune_model.py` quantizes one model both ways in a single run and
prints the two perplexity penalties side by side, which is the measurement that
actually matters. [RESULTS.md](RESULTS.md) has what those come out to on three
models, against the reference implementation.

---

## Using the reference implementation from Python

Everything below runs in a plain REPL, on any machine, with no model.

### Quantize a weight matrix

```python
import numpy as np
from mround.reference.quantize import fake_quantize, init_params
from mround.schemes import QuantScheme

weight = np.random.default_rng(0).normal(size=(8, 64))
scheme = QuantScheme(bits=4, group_size=32)

result = fake_quantize(weight, init_params(weight, scheme), scheme)
result.qdq  # the reconstruction, same shape as weight
result.codes  # integer codes, in [-8, 7] at 4 bits symmetric
result.scale  # per-group scale, and note: it is signed
```

That last point surprises people, so it is worth pausing on. Under a symmetric
scheme the scale carries the sign **opposite** to the group's dominant extreme.
That is not a bug and not something to normalize away: it maps the dominant
extreme onto the code `-2**(bits-1)`, the one extra code that the two's
complement range provides, so the largest weight in each group reconstructs
exactly instead of clipping. Removing the sign inversion costs 50 percent of
that weight's magnitude at 2 bits. DOCUMENTATION.md section 5.2.1 works through
it.

### Learn the rounding

Learning needs activations, because the whole point is to optimize against the
layer's output rather than against its weights. The example script has a helper
that makes plausible correlated ones:

```python
import sys

sys.path.insert(0, "examples")
from first_quantization import synthetic_layer
from mround.reference.tuning import tune_layer
from mround.schemes import TuningConfig

weight, activations = synthetic_layer(seed=0)

result = tune_layer(weight, activations, scheme, TuningConfig(iters=200))
result.initial_loss  # exactly what round-to-nearest achieves
result.final_loss  # after learning
result.improvement  # the fraction saved
result.losses  # the whole trajectory, if you want to plot it
```

`initial_loss` is the round-to-nearest baseline rather than an arbitrary
starting point, because the initial parameters reproduce round-to-nearest
exactly. That is what makes `improvement` an honest number.

### Pack codes for storage

```python
from mround.reference.packing import pack_codes, unpack_codes

codes = np.random.default_rng(1).integers(0, 8, size=(4, 64))
packed = pack_codes(codes, bits=3)  # (4, 6) uint32: 64 x 3 bits = 6 words
np.array_equal(unpack_codes(packed, 64, 3), codes)  # True
```

Widths that divide 32 (2, 4, 8) pack at fixed offsets within a word. The others
(3, 5, 6, 7) form a contiguous little-endian bit stream where elements straddle
word boundaries, so exactly 32 codes occupy exactly `bits` words with nothing
wasted.

### Allocate bits under a size budget

```python
from mround.planner.allocator import LayerOption, allocate_bits

options = {
    "attn": [LayerOption(2, 2048, 4.0), LayerOption(4, 4096, 1.0), LayerOption(8, 8192, 0.1)],
    "mlp": [LayerOption(2, 8192, 9.0), LayerOption(4, 16384, 2.0), LayerOption(8, 32768, 0.2)],
}
allocate_bits(options, budget_bits=26000).by_layer  # {'attn': 8, 'mlp': 4}
```

Each `LayerOption` is `(bits, cost_bits, delta_loss)`. The allocator solves a
multiple-choice knapsack exactly, so this is the true optimum rather than a good
guess. Raising the budget shifts the answer in ways a greedy rule would miss:

```
budget 20480 -> {'attn': 4, 'mlp': 4}   4.00 bits average
budget 26000 -> {'attn': 8, 'mlp': 4}   4.80 bits average
budget 40000 -> {'attn': 4, 'mlp': 8}   7.20 bits average
```

That third row is the interesting one. With more budget the allocator abandons
8 bit attention in favour of 8 bit MLP, because the MLP is where the loss
actually is. A greedy upgrade rule would never walk an earlier choice back.

---

## Where things live

```
mround/
├── reference/     the NumPy implementation: the oracle, works anywhere
├── core/          the MLX implementation, parity-checked against it
├── planner/       bit allocation and sensitivity scoring; importance.py is a stub
├── pipeline/      loading, calibration, and the block loop
├── formats/       MLX export works; GGUF and Core ML are stubs
├── kernels/       custom Metal: empty by design, none justified yet
├── eval/          perplexity works; zero-shot tasks are a stub
└── cli/           quantize, eval, doctor, version

examples/          runnable demonstrations
scripts/           probes, the environment capture, the overlap check
benchmarks/        the measurement ledger and its environment manifests
tests/unit/        needs nothing: no model, no MLX, no Mac
tests/parity/      planned: a README only, the corpus is not built
tests/integration/ planned: empty
```

A stub is a module whose functions raise `NotImplementedError` but which carries
the full type signatures and documentation for what it will do, marked by a
`Status:` line in its docstring that a test keeps honest against the code.

---

## Running tests

```bash
pytest tests/unit -q                    # everything, about 20 seconds
pytest tests/unit -q -m "not slow"      # the fast suite, about 6 seconds
pytest tests/unit/test_reference_tuning.py -v    # does the method work?
pytest tests/unit/test_allocator.py -v           # exact versus brute force
```

Tests are organized by what they need rather than by what they cover. Anything
in `tests/unit` needs nothing at all. Tests that need MLX, a real model or the
reference corpus mark themselves and skip when their prerequisites are absent,
so a fresh clone has a green suite rather than a wall of errors.

Two test files are worth reading as documentation in their own right.
`test_reference_quantize.py` contains a finite difference check of every
analytic gradient, and the helper that builds the comparison function is
probably the clearest statement anywhere here of what a straight-through
estimator actually computes. `test_allocator.py` compares against exhaustive
search, including an instance constructed so that greedy allocation is provably
wrong.

---

## What to read, in what order

| If you want to | Read |
|---|---|
| Understand the project | [README.md](README.md) |
| See what it scores, and against what | [RESULTS.md](RESULTS.md) |
| Implement or change any numerics | [DOCUMENTATION.md](DOCUMENTATION.md) sections 1 and 5 |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

Section 7 of DOCUMENTATION.md is the one to read before debugging anything
numerical. It lists the hazards that have actually cost this project time, and
three of them look like edge cases and are not.
