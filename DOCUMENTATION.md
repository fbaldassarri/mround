# MRound Technical Documentation

**Version: 0.0.1. Last updated: 2026-09-17.**

This document specifies what MRound does and how it is built. It is the
authoritative technical reference; where it disagrees with a comment in the
code, this document is wrong and should be corrected, or the code is wrong and
should be fixed, but the disagreement must be resolved rather than tolerated.

Citations of the form `MEMORY.md D-0NN`, here and in the source comments, refer
to MRound's decision log: what was decided, when, and which measurement decided
it. That log is part of the development repository and is not published. The
citations are kept because each one identifies a specific recorded decision
rather than gesturing at history, and because the reasoning they point at is
usually reconstructible from the code and the tests around them.

**Contents**

1. The method
2. Why Apple Silicon changes the engineering
3. Architecture
4. Module layout
5. Numerical specification
6. Formats
7. MLX implementation hazards
8. Validation methodology
9. The Neural Engine, honestly
10. Glossary

---

## 1. The method

### 1.1 The problem with rounding

Weight-only quantization stores a weight matrix at reduced precision and
reconstructs an approximation at inference time. The naive approach, round to
nearest, chooses for each weight the representable value closest to it. This is
optimal for each weight considered alone and suboptimal for the matrix
considered as a whole, because what matters is not the error in any weight but
the error in the layer's output. Rounding one weight up and its neighbor down
may cancel; rounding both up may compound.

The search space is enormous, so the question is how to search it tractably.
SignRound's answer: make the rounding decision continuous and differentiable,
then optimize it against the thing you actually care about, which is the block's
output.

### 1.2 The fake-quantization operator

For a weight matrix `W` partitioned into groups along the input dimension, with
`b` bits:

```
    s   = f(max(W_g), min(W_g), alpha, beta)      per-group scale
    q   = clamp( round_ste( W / s + v ), q_min, q_max )
    W'  = s * q                                   symmetric
    W'  = s * (q - z)                             asymmetric
```

Three quantities are learned:

- `v`, a per-weight rounding perturbation, initialized to zero and **constrained
  to `[-0.5, 0.5]`**. That bound is what makes this *learned rounding* rather
  than unconstrained weight learning: within it, each weight can be moved to
  either of the two representable values adjacent to it, and no further. Without
  the bound the method silently becomes something else.
- `alpha` and `beta`, per-group clipping coefficients, initialized to one, which
  scale the observed maximum and minimum before the scale is computed. Reducing
  the range increases resolution for the bulk of the distribution at the cost of
  clipping outliers.

`round_ste` is rounding with a straight-through gradient: the forward pass
rounds, the backward pass passes the gradient through unchanged, as if the
operation were the identity. Without this, the gradient is zero almost
everywhere and nothing can be learned.

`clamp` needs its own specification for the same reason, and it is easy to miss.
The value reaching the clamp has already been rounded, so it is an integer, and a
code landing exactly on `q_min` or `q_max` is therefore the ordinary case rather
than a coincidence: 43 percent of weights at 2 bits, 7 percent at 4 bits, none at
8. **The boundary counts as inside.** A code exactly on `q_min` or `q_max`
receives gradient; only a code the clamp actually moved does not. See section 5.7
for why, and MEMORY.md D-013 for the failure that made the point.

### 1.3 The objective

Quantization proceeds one transformer block at a time. For each block, the
original outputs are computed and cached, the block's weights are wrapped in the
fake-quantization operator, and the learnable quantities are optimized to
minimize the discrepancy between the quantized block's output and the cached
original:

```
    L = mean( ( Q(x) - F(x) )^2 )
```

computed in float32 over the calibration batch, masked by the attention mask
where padding is present. This is layer-wise reconstruction: no labels, no
task loss, no full backward pass through the model. Only the current block's
parameters receive gradients, which is what makes the method cheap relative to
quantization-aware training.

### 1.4 Signed gradient descent

The optimizer uses only the sign of the gradient:

```
    p  <-  p - lr * sign( dL/dp )
```

Every parameter moves by exactly the learning rate at every step, in the
direction the gradient indicates. The learning rate decays linearly to zero over
the step budget, which makes the trajectory a coarse-to-fine search: large
strides early, progressively finer adjustments later, terminating at zero.

This suits the problem. The loss surface over rounding perturbations is
pathological, full of flat regions where the rounding does not change and cliffs
where it does. Gradient magnitudes carry little reliable information about step
size. The sign carries the useful part.

Two consequences follow directly and are worth stating explicitly, because both
have implementation implications:

**The update is invariant to positive rescaling of the loss.** Multiplying `L`
by any positive constant leaves the trajectory unchanged. Loss scaling is
therefore purely a numerical guard against gradient underflow in reduced
precision, and is unnecessary when gradients are accumulated in float32.

**Parameters must be projected back into their valid ranges after each step**,
because the update rule itself respects no constraints. This applies to `v`, which
must be clipped to `[-0.5, 0.5]`, and to the clipping coefficients, which have
their own bounds. Nothing downstream enforces either bound: the clamp in the
quantization operator constrains `q`, not `v`, so an unprojected `v` will drift
past a full integer code and change the algorithm.

### 1.5 What v2 adds

**Searched scale initialization.** Rather than starting from clipping
coefficients of one, v2 searches a grid of candidate scales per group,
minimizing importance-weighted squared reconstruction error, and starts the
optimization from the winner. Because signed gradient descent takes fixed-size
steps from wherever it starts, a better starting point is worth more here than
it would be with an adaptive optimizer.

**Importance weighting.** An importance matrix, accumulated as the sum of
squared activations per input channel over the calibration set, weights the
reconstruction error so that weights multiplying large activations are
approximated more carefully. A weight that is always multiplied by near-zero
activations does not deserve precision.

**Outlier-suppressed loss.** At very low bit widths, a small number of
enormous errors dominate the mean squared error and destabilize the optimization.
Excluding the largest errors from the objective stabilizes tuning. The exact
definition of "excluding" matters and is specified in section 5.

**Sensitivity scoring.** A first-order estimate of how much each layer's
quantization contributes to the overall loss increase, combining gradient
magnitude with the actual perturbation quantization introduces:

```
    DeltaLoss = || g ⊙ (W_q - W_f) ||_1
```

The motivation is first order. To first order the loss increase from perturbing
weights is `g · ΔW`, the gradient dotted with the perturbation. DeltaLoss takes
absolute values before summing, so it is not that quantity: it is an upper bound
on its magnitude, by the triangle inequality, and it is better described as a
cancellation-free proxy for the first-order term. The distinction matters,
because a signed sum lets opposing perturbations cancel and can report a
near-zero score for a layer that is being badly damaged in both directions.

Two caveats worth carrying. Near a minimum of the training loss the first-order
term is small by construction, which is why classical saliency measures reach for
curvature instead. And the claim that this scores layers better than a curvature
proxy is the v2 paper's, not yet MRound's; it is a hypothesis this project should
test rather than an established result.

How the score is obtained is part of the specification, because each item
changes the numbers. The gradients come, by default, from one backward pass
per batch taken at the widest candidate width, and every width's perturbation
is scored against them; this is a measured divergence from the reference
(MEMORY.md D-034), because the reference's procedure, a backward per candidate
on the model quantized at that width, sends the low widths' gradients through
a network their own grid has destroyed and measurably starves the late blocks.
The reference procedure remains available as the ``own_width`` control, where
the gradient is evaluated at each width's own quantized operating point. In
either mode the loss is the mean shifted
causal cross-entropy, labels being the inputs themselves, on a small
calibration draw whose defaults are 128 sequences of 1024 tokens, the budget
every published mixed measurement used (D-031 amended; the reference's own 16
by 256 was the default until that budget measured 4.70 percent better on the
same model and seed); `quantize` and `plan_mixed_precision` take the draw as
`scoring_samples` and `scoring_seq_len`, and the reference now warns that its
own former default is too small once a 2 bit option is on the menu, so the
published mixed 2.5 head to heads were scored at 128 by 1024 on both sides.
The draw is fed to the backward in batches holding a constant 2048 tokens
(`SENSITIVITY_BATCH_TOKENS`: 8 sequences at 256 tokens, 2 at 1024, 1 at 2048,
never more than the draw), which is what keeps scoring memory flat as the
sequence grows; the batch can be overridden with `scoring_batch_size` to
reproduce a run measured at another batching. Scores sum over batches, so a
`predicted_loss` is comparable only between runs batched the same way, while
the allocation itself is invariant to any positive rescaling of the scores
(MEMORY.md D-042). The layers scored are exactly the layers the block loop tunes;
embeddings and everything outside the blocks are excluded from scoring. For
weight-only schemes the activation term the reference conditionally adds is
identically absent, so the score is the weight term alone. Scores are raw sums
with no normalization: both the score and the storage cost scale with layer
size, so the allocator's implicit loss-per-bit trade stays dimensionally
consistent across layers. One deliberate divergence: under a searched scheme
the quantize-dequantize being scored uses the searched grid, because the score
exists to predict the damage the actual quantizer will do and MRound's low-bit
quantizer is the searched one. MEMORY.md D-031 records these decisions.

**Mixed-precision allocation.** Given a sensitivity score per layer per
candidate bit width, and a storage cost per choice, choose one bit width per
layer minimizing total predicted loss subject to a budget on total size. This is
a multiple-choice knapsack problem, solved exactly by dynamic programming over
the reachable integer costs with Pareto pruning; an earlier draft specified a
discretized budget axis, and MEMORY.md D-012 records why exact integer state
replaced it. The storage cost of a choice counts the packed codes and the scale
metadata, the same accounting the export path reports. Measured cost: a few
hundred layers allocate in seconds against a tuning run that costs hours.

---

## 2. Why Apple Silicon changes the engineering

The method is hardware-independent. The engineering around it is not, and the
differences are large enough to justify a separate implementation rather than a
port.

**Unified memory removes the transfer problem and replaces it with a ceiling
problem.** Implementations targeting discrete accelerators spend real complexity
on staging: moving blocks to the device, offloading activations to host memory,
recomputing rather than storing. On Apple Silicon there is one pool. The
transfer logic is not optimized, it is deleted. What replaces it is a single
hard ceiling shared with everything else on the machine, which makes the memory
discipline within a block more important, not less.

**The memory ceiling is high, and this is the project's central opportunity.** A
128 GB Mac has more memory available to a quantization run than most single
discrete GPUs. Because quantization is block-wise, peak usage is governed by the
largest block plus cached activations rather than by the whole model. The
combination means a Mac can plausibly quantize models that require multi-GPU
setups elsewhere. This claim is the project's headline and it will be established
by measurement in Phase 5, not asserted.

**Compute is different in character.** The GPU is well suited to the elementwise
and reduction work that dominates the tuning loop. The scale search in
particular is embarrassingly parallel across groups. But Apple GPUs have less
raw throughput than datacenter parts, so the correct expectation is that MRound
quantizes larger models than a consumer discrete GPU can, not that it quantizes
them faster than a datacenter GPU does. The value proposition is capability and
accessibility, not peak speed.

**The framework is lazily evaluated and functional.** MLX builds a graph and
evaluates on demand, and its arrays are immutable. This is a genuine
architectural difference from eager frameworks with mutable tensors, and it
shapes the code more than any other single factor. Section 7 covers it in
detail.

**A third engine exists that cannot be used for this.** The Neural Engine is
efficient and irrelevant to quantization. Section 9 explains why, and what it
might still be good for.

---

## 3. Architecture

Five layers, each testable in isolation. The ordering reflects dependency, not
importance.

```
   ┌──────────────────────────────────────────────────────────┐
   │  Interface        CLI, Python API, configuration files    │
   ├──────────────────────────────────────────────────────────┤
   │  Pipeline         model loading, block discovery,         │
   │                   calibration, orchestration              │
   ├──────────────────────────────────────────────────────────┤
   │  Planner          importance matrix, sensitivity scoring, │
   │                   dynamic-programming bit allocation      │
   ├──────────────────────────────────────────────────────────┤
   │  Optimizer        fake quantization, scale search,        │
   │                   signed gradient descent, block loss     │
   ├──────────────────────────────────────────────────────────┤
   │  Kernels          MLX operations, custom Metal where      │
   │                   profiling justifies it                  │
   └──────────────────────────────────────────────────────────┘
```

Each layer maps to exactly one package, and three packages sit outside the
stack:

| Layer | Package |
|---|---|
| Interface | `mround/cli/` and `mround/api.py` |
| Pipeline | `mround/pipeline/` |
| Planner | `mround/planner/` |
| Optimizer | `mround/core/` |
| Kernels | `mround/kernels/` |
| *(cross-cutting)* | `mround/reference/`, the NumPy implementation of the same mathematics as `core/`, depending on nothing above it |
| *(cross-cutting)* | `mround/formats/`, which serializes what the stack produces and depends only on Kernels |
| *(cross-cutting)* | `mround/eval/`, which consumes finished checkpoints and depends on nothing above |

Weight packing lives in `formats/` rather than `kernels/`, because it is part of
a checkpoint's definition rather than part of the compute path. A custom Metal
kernel that accelerates packing would live in `kernels/` and be called from
`formats/`.

`mround/reference/` deliberately duplicates the mathematics of `core/` in
NumPy. That duplication is the point rather than an oversight: it gives the MLX
implementation a local oracle that shares no code with it, so a disagreement
between the two is evidence about the port rather than about a shared bug. It
also runs on any machine, which is what keeps the numerical core testable in
continuous integration without Apple hardware. It must never import MLX, and a
test enforces that. When the two implementations disagree, the reference is
presumed right until shown otherwise, because it is the one with exhaustive
tests and finite-difference-verified gradients.

**The optimizer layer knows nothing about transformers.** It quantizes a weight
matrix given inputs and target outputs. This makes it unit-testable on synthetic
data with no model loaded, which is what makes fast iteration possible.

**The planner layer knows nothing about MLX.** It consumes sensitivity scores
and costs and produces an allocation. It is testable with hand-computed
examples, including adversarial ones where greedy allocation is provably
suboptimal.

**The pipeline layer holds all model-specific knowledge**, so adding an
architecture touches one layer.

---

## 4. Module layout

```
mround/
├── reference/             NumPy: the executable spec and the MLX oracle
│   ├── quantize.py        fake quantization with analytic gradients
│   ├── scale_search.py    v2 candidate scale grid search, the oracle
│   ├── optimizer.py       signed gradient descent with linear decay
│   ├── losses.py          reconstruction losses, outlier suppression
│   ├── packing.py         uint32 bit packing for all widths
│   └── tuning.py          the single-layer learned-rounding loop
├── core/
│   ├── quantizer.py       fake quantization, scale computation, STE
│   ├── signsgd.py         signed gradient descent with linear decay
│   ├── scale_search.py    v2 candidate scale grid search
│   ├── losses.py          reconstruction losses, outlier suppression
│   └── tuning.py          the single-layer learned-rounding loop
├── planner/
│   ├── importance.py      importance matrix accumulation
│   ├── sensitivity.py     DeltaLoss definition, storage costs, option assembly
│   └── allocator.py       multiple-choice knapsack via DP
├── pipeline/
│   ├── loader.py          model loading, architecture detection
│   ├── blocks.py          block discovery, activation capture
│   ├── calibration.py     dataset handling, batching, masking
│   ├── scoring.py         whole-model DeltaLoss backward per candidate width
│   ├── device.py          peak memory as the array framework reports it
│   └── runner.py          orchestration, memory discipline
├── formats/
│   ├── packing.py         uint32 bit packing for all widths
│   ├── mlx_export.py      MLX-native checkpoints
│   ├── gguf_export.py     GGUF for llama.cpp
│   └── coreml_export.py   Phase 7, research
├── kernels/
│   └── metal/             custom kernels, each with its justifying profile
├── eval/
│   ├── perplexity.py
│   ├── zeroshot.py
│   └── results.py         run ledger: every measurement, machine-readable
├── cli/
│   └── main.py            command-line entry point
├── api.py                 the public Python API, the only stable surface
├── schemes.py             QuantScheme and TuningConfig, shared by every layer
└── exceptions.py          the error hierarchy

tests/
├── unit/                  no model required, synthetic data
├── parity/                against the dumped reference corpus
├── integration/           end-to-end on small models
└── fixtures/              the parity corpus and calibration sets

scripts/
└── check_module_layout.py verifies this diagram against the tree
```

`schemes.py` and `exceptions.py` sit at the package root rather than inside a
layer because every layer depends on them. `schemes.py` in particular imports
nothing from the rest of the package and nothing from MLX, which is what lets
the planner and the formats layer both use it without either taking on the
other's dependencies. A layer-specific home for it would create an import cycle
the moment two layers needed it, which is immediately.

The diagram above is checked, not trusted. `scripts/check_module_layout.py`
parses it and compares it against the tree in both directions: a documented
module that does not exist is a broken promise, and an undocumented module is a
layer someone added without saying so. It runs in continuous integration and as
a pre-commit hook.

---

## 5. Numerical specification

Everything in this section is a decision that changes results. Each is stated
explicitly so that a discrepancy can be traced to a choice rather than to a
mystery. Items marked *to be determined* are open, and carry the phase in which
they will be settled; they are the only permitted gaps, and filling one silently
is a defect.

### 5.1 Precision policy

Weights load in float16 or bfloat16 as stored. The fake-quantization forward
computes in float32. Loss accumulates in float32 always, without exception; the
reconstruction errors are small and float16 accumulation loses them. Learnable
parameters are float32.

Exported scales and biases follow the source model's floating dtype: a bfloat16
model exports bfloat16 scales, a float16 model exports float16 scales. This
avoids producing a mixed-dtype checkpoint, which is what casting everything to
float16 unconditionally would do. Where a target format mandates a specific
scale dtype, the exporter for that format converts and records the conversion in
the checkpoint metadata.

Because signed gradient descent uses only the sign, no loss scaling is applied.
This is a deliberate divergence from implementations that must guard against
float16 gradient underflow.

**Float32 accumulation is a policy, not a guarantee the framework provides.**
Measured on Apple Silicon with MLX 0.32, `mx.matmul` on float32 operands returns
a relative error of 8.5e-4, roughly four orders of magnitude worse than float32.
It is the GPU matrix-product kernel specifically: the same product on the CPU
stream is accurate to 2.3e-7, and an elementwise multiply followed by a sum on
the same GPU is accurate to 8.2e-8. The kernel rounds its inputs rather than
accumulating narrowly, which is established by the error being flat from a
reduction length of 16 to one of 4096, so the loss is a fixed relative amount and
does not compound with model width.

Everything above still holds as policy. What changes is that any expression which
subtracts two nearly equal matrix products must be rewritten so that it does not,
because that error is proportional to the large product and the subtraction hands
all of it to the small difference. Section 5.8 states the rule; MEMORY.md D-015
records the measurement and D-014 the consequence.

### 5.2 Scale computation

Both branches first form the clipped group extremes, forcing zero into the range
so that the zero point is always representable:

```
    w_max = max( max(W_g), 0 ) * alpha
    w_min = min( min(W_g), 0 ) * beta
```

Forcing zero into the range matters for groups whose weights are all one sign.
Without it the asymmetric zero point falls outside the code range and no format
that stores an integer zero point can represent the group.

**Symmetric.** The scale carries the sign *opposite* to the larger-magnitude
extreme. That inversion is not a quirk; it is the mechanism that makes the
asymmetric code range work, and section 5.2.1 explains why.

```
    dominant = max( w_max, |w_min| )
    s     = -dominant / 2^(b-1)   if  w_max >= |w_min|
          = +dominant / 2^(b-1)   otherwise
    s     = clamp_magnitude( s, eps )
    q     = clamp( round_ste( W/s + v ), -2^(b-1), 2^(b-1) - 1 )
    W'    = s * q
```

`clamp_magnitude(s, eps)` moves `s` away from zero to at least `eps` in
magnitude while preserving its sign, which is why `s` must be allowed to be
negative: a magnitude clamp on a signed quantity cannot be a plain minimum. Zero
has no sign, so the degenerate case is specified rather than left to the
implementation:

```
    clamp_magnitude(s, eps) = +eps                      if s == 0
                            = sign(s) * max(|s|, eps)   otherwise
```

Without the first branch, an all-zero group yields `s = 0` and reintroduces
exactly the division by zero that `eps` exists to prevent.

#### 5.2.1 Why the scale sign is inverted

The code range `[-2^(b-1), 2^(b-1) - 1]` is the standard two's-complement range.
It wastes no code point, but it is asymmetric: there is one more code available
on the negative side than on the positive side.

The sign inversion spends that extra code on whichever side of the distribution
actually needs it. When the positive side dominates, the scale is negative, so
the largest positive weight maps to `-2^(b-1)`, which is exactly representable.
When the negative side dominates, the scale is positive and the largest negative
weight maps to `-2^(b-1)` for the same reason.

**The dominant extreme therefore reconstructs with zero error and never clips.**
Verified numerically at 2, 4, and 8 bits:

| Case | scale sign | dominant extreme | clips |
|---|---|---|---|
| positive side dominates | negative | reconstructs exactly | no |
| negative side dominates | positive | reconstructs exactly | no |
| exact magnitude tie | negative | positive side exact | the negative side clips by one code |

Only an exact magnitude tie causes a clip, and only on the non-dominant side.
For real weight distributions that is a measure-zero case.

The naive alternative, an unsigned `s = max(|W|) / 2^(b-1)`, clips the dominant
extreme by one code: a 0.8 percent magnitude shortfall at 8 bits, 12.5 percent at
4 bits, and 50 percent at 2 bits. The other naive alternative,
`s = max(|W|) / (2^(b-1) - 1)`, clips nothing but leaves one code permanently
unused and lowers resolution for every weight in the group.

The signed construction is strictly better than both, which is why MRound adopts
it. Two consequences follow and both must be honored downstream:

- `s` is genuinely signed, so `clamp_magnitude` must preserve its sign and a
  plain minimum clamp is wrong.
- Exported scales will be negative for roughly half of all groups. Any consumer
  that assumes positivity, or that takes an absolute value, corrupts the weights.
  Section 6.2 covers what this means for export.

This corrects an earlier draft of this specification, which described the
dominant extreme as clipping and raised that as an open question. It does not
clip. See MEMORY.md D-009.

**Asymmetric.**

```
    s = (w_max - w_min) / (2^b - 1)
    s = clamp( s, min=eps )
    z = round_ste( -w_min / s )
    q = clamp( round_ste( W/s + v ) + z, 0, 2^b - 1 )
    W' = s * (q - z)
```

Here `s` is non-negative by construction, so a plain minimum clamp is correct.
Because zero was forced into the range, `z` lies in `[0, 2^b - 1]`. The zero
point passes through the straight-through estimator and therefore participates
in gradients.

`eps` guards against division by zero for all-zero groups. Its value is `1e-8`
for float32 scales; `1e-5` is reserved for reduced-precision scales
(`DEFAULT_EPS_LOW_PRECISION`), which no code path uses yet since every scale
is computed in float32.

**Under v2's searched initialization the scale is parameterized differently.**
The search produces `s_init` per group, and the clipping coefficient becomes a
multiplier on that result rather than on the observed range:

```
    s = s_init * beta             symmetric, v2 with searched initialization
```

`alpha` is unused in this form (it is carried, projected, and receives a
gradient of exactly zero; MEMORY.md D-030 and a unit test pin its inertness),
and `beta` is centered on 1.0 meaning "accept the search result" rather than
"no clipping". This mirrors the reference, whose searched branch reads its
upper coefficient and ignores the lower one. This is why the permitted range
differs between the two parameterizations in section 5.3, and it is the reason
the v2 range must extend above 1.0: otherwise the optimizer could only ever
shrink the searched scale, never grow it.

### 5.3 Parameter ranges

| Parameter | Init | Range | Granularity |
|---|---|---|---|
| `v` | 0 | `[-0.5, 0.5]` | per weight |
| `alpha`, `beta` | 1.0 | `[0.1, 1.0]` in v1 | per group |
| `alpha`, `beta` | 1.0 | `[0.5, 1.5]` in v2 with searched initialization | per group |

All three are projected into range after every optimizer step. Nothing else
enforces these bounds.

The lower bound of `0.1` on the clipping coefficients is a guard, not a tuning
choice: at exactly zero the group range collapses, the scale falls to `eps`, and
every weight in the group saturates to a single code. A strictly positive floor
makes that state unreachable.

The v2 range differs because the coefficients mean something different there.
With searched initialization active `beta` multiplies the *searched* scale
rather than the observed range (and `alpha` is inert, section 5.2), so 1.0 is
the search result rather than "no clipping", and the useful adjustments run in
both directions around it. The `[0.5, 1.5]` bound is MRound's own choice: the
reference's searched path bounds its coefficient at `[0.0, 2.0]`, and the
narrower range has not been A/B tested against it (audit of 2026-09-17). Applying the v1
range to the v2 parameterization would forbid the optimizer from ever increasing
the scale. This coupling between initialization mode and parameter range is the
single most confusing part of the specification and is stated here for that
reason.

### 5.4 Learning rate

The learning rate is derived from the step budget so that the total distance a
parameter can travel is independent of how many steps are taken:

```
    lr = c / iters        c = 1.0  for b >= 4
                          c = 2.0  for b < 4
```

Since every step moves a parameter by exactly `lr` and the rate decays linearly
to zero, the total possible excursion is about `c/2`. For `b >= 4` that is 0.5,
exactly the width of `v`'s permitted range in one direction, which is the
property that makes this scaling the right one. Low bit widths get the larger
constant because they need to explore more of that range.

Decay is linear from `lr` to exactly zero over the budget, stepped once per
iteration. The clipping coefficients share `v`'s learning rate; the reference
accepts a separate rate for them (`minmax_lr`) and MRound's `TuningConfig`
carries the field but refuses any value other than `None`, because no tuning
loop consumes it yet, and the same holds for `gradient_accumulate_steps`
other than 1.

**Which parameters a layer keeps.** The loop measures the loss with the
current parameters, records them if that measurement is the lowest so far, and
only then applies the update. The parameters a layer keeps are therefore the
ones behind the lowest measurement, `best_step` counts the updates that
produced them, and the parameters left by the final update are never measured
and can never be kept. Step zero measures the untouched parameters, so
`initial_loss` is `losses[0]` and needs no forward of its own. This is the
reference implementation's semantics (its block loop evaluates, records the
best, then steps, and its `init_loss` is the loss at iteration zero), adopted
rather than "improved": scoring the final update would make MRound keep
parameters the reference never considers, for no gain the method describes.
The reference also offers a switch to keep the last step's parameters
regardless; MRound does not expose one. MEMORY.md D-048.

Two divergences from the reference in this rule are deliberate and recorded.
The reference applies `c = 2.0` only when `iters >= 1000` as well as
`b <= 3`; MRound applies it at every step budget below 4 bits, so at the
standard 200 iterations a 2 bit run here starts at `lr = 0.01` where the
reference starts at `0.005` (MEMORY.md D-010 and the audit entry of
2026-09-17). And in a mixed precision run the block loop resolves `b` from the
run's base `bits`, once per block, rather than per layer from the width the
planner assigned; every published mixed run was produced that way and the
choice is recorded as D-043 rather than changed under them.

Default step budget: 200 iterations per block, batch size 8, sequence length
2048, 128 calibration samples; group size 64 everywhere, in every measured
run and in `QuantScheme`, `quantize`, `quantize_round_to_nearest`,
`plan_mixed_precision` and the command line, a default settled on generality
rather than quality (MEMORY.md, 2026-09-20: every dimension divisible by 128 is
divisible by 64 and the converse fails, so 128 leaves most of SmolLM2 dense).
The reference's own higher-quality recipe, 1000 iterations over 512 samples,
has been measured on both sides at mixed 2.5 on Qwen2.5-0.5B (ledger row 48):
MRound 22.3231 against the reference's 25.8199, a margin that narrows from
19.22 to 13.54 percent and holds, while MEMORY.md D-028 records that five
times the steps on a fixed small corpus is harmful.

### 5.5 Outlier-suppressed loss

At bit widths below 4, or when the configuration requests it, the largest
absolute errors are excluded from the objective. The specification, which must
be honored exactly because each choice changes the gradient:

- The excluded fraction is 0.1 percent of elements, floored at one: the count
  is `max(1, floor(n / 1000))`, so at least one element is always excluded
  however small the batch. This is the reference's own rule
  (`topk = max(1, int(numel / 1000))`); below 1,000 elements it excludes one
  element where 0.1 percent would exclude none, which only a synthetic layer
  or a unit test ever sees (MEMORY.md D-048).
- Selection is over the flattened batch, not per row. The reference and the
  NumPy implementation exclude exactly that many elements; the MLX
  implementation selects by a threshold comparison, because MLX has no boolean
  mask assignment, and excludes every element tied at the threshold, a
  documented and measure zero deviation on real residuals.
- Excluded elements are zeroed in the numerator; the denominator remains the
  full element count. Excluded elements therefore dilute the mean rather than
  being removed from it.
- The reduction is always the mean, including under gradient accumulation.

The third point is counterintuitive and easy to get wrong in either direction.
It is specified here so the implementation is a decision rather than an accident.

### 5.6 Activation source

When quantizing block N, two activation streams run at once, which is
Q-001's answer (MEMORY.md, closed by reading the reference; D-026 records the
implementation). The optimization *target* for block N is block N run with its
original weights on the full precision stream, so the target chain equals the
original model's activations. The *input* fed to block N during tuning comes
from the already quantized blocks 1 through N-1, which is the reference's
default for SignRound and `api.quantize`'s default here
(`quantized_inputs=True`); turning it off halves activation memory and is
exposed for that measurement. Both streams are stored in float32 regardless of
the model's width, because the block objective is a difference of two block
outputs and a bfloat16 stream would put a noise floor of the same order as the
signal on one side of it (D-026).

### 5.7 The clamp boundary in the backward pass

A code exactly on `q_min` or `q_max` counts as inside the clamp and receives
gradient. The mask is `q_min <= raw <= q_max`, inclusive at both ends, where
`raw` is the rounded value before clamping.

This is a decision, not a derivation. The straight-through surrogate being
differentiated is `clamp(W/s + v + c, q_min, q_max)` with the rounding residual
`c` held fixed, its clamp argument sits exactly at the kink whenever the code is
on the boundary, and the subderivative of `clamp` there is the whole interval
`[0, 1]`. Both conventions are legitimate.

Inclusive is specified for two reasons. It is what the reference implementation
does, since that reaches the clamp through PyTorch's `clamp`, which is inclusive
at both ends. And it is the better of the two on its own merits: a code on
`q_max` is not saturated in both directions, so passing the gradient keeps the
downward direction available, whereas blocking it freezes the weight at
round-to-nearest for the entire optimization. A step in the direction the clamp
does block is absorbed by the forward clamp and costs one wasted step, which is
recoverable; a frozen parameter is not.

The practical stake is larger than it sounds, because the clamp's input is an
integer and the boundary is therefore hit constantly rather than rarely: at 2
bits, the convention decides whether 43 percent of the rounding perturbations
learn anything at all. Implementations must supply it explicitly. Frameworks
disagree, and inheriting whichever one a framework happens to implement is how
this becomes a silent divergence. MEMORY.md D-013 records the run where it was.

### 5.8 Never form a small residual by subtracting two large products

Where an error signal is the difference between a quantized output and a
full-precision one, it is computed by multiplying the *difference of the
operands* wherever the algebra permits:

```
    r  =  x @ (Wq - W).T                 required
    r  =  x @ Wq.T  -  x @ W.T           forbidden where the first form exists
```

The two are algebraically identical and numerically are not close. `mx.matmul`
returns a relative error of 8.5e-4 on float32 operands (section 5.1). That error
is proportional to the product being computed, so the second form gives the
residual an error of 8.5e-4 times the *full* output, while the residual itself is
smaller than the output by a factor that grows with bit width. Measured on a
64-wide layer: the perturbation is 0.5 percent of the residual at 2 bits and 20
percent at 8. The first form keeps the error proportional to the residual, so it
stays at 8.5e-4 regardless of bit width.

Twenty percent noise on the quantity being minimized is not a precision detail.
Signed gradient descent consumes only the sign of the gradient, so noise at that
level flips signs directly and the optimizer follows it.

The rule has a boundary, and it must be stated because Phase 2 runs straight into
it. It applies wherever the difference can be pushed through to the operands,
which holds for a single linear layer and does not hold for a block objective
with nonlinearities between the two outputs being compared. There the subtraction
is unavoidable, and section 5.9 gives the remedy.

### 5.9 Split-precision products, where the subtraction cannot be removed

Where section 5.8's rewrite is unavailable, each of the two products is computed
by splitting both operands and summing three products:

```
    A_hi = bfloat16(A)      A_lo = A - A_hi
    B_hi = bfloat16(B)      B_lo = B - B_hi

    A @ B  ~=  A_hi @ B_hi  +  A_hi @ B_lo  +  A_lo @ B_hi
```

The high halves survive the kernel's input rounding exactly, because bfloat16 is
narrower than the precision the kernel keeps, and the two cross terms carry back
what the rounding dropped. The omitted low-times-low term is what remains, and it
is negligible.

**Three products. Two is worse than one.** Measured: one product 8.5e-4, two
products 1.6e-3, three products 3.1e-6. Dropping either cross term leaves the
whole of one operand's low half unaccounted for, which is larger than the error
being corrected. A two-product form is not a cheaper approximation of this; it is
wrong, and it is wrong in a way that looks like a reasonable economy.

On the layer residual, and therefore on the quantity the optimizer minimizes:

| bits | one product | three products |
|---|---|---|
| 2 | 1.16e-3 | 1.16e-5 |
| 3 | 1.57e-3 | 2.96e-5 |
| 4 | 3.31e-3 | 5.88e-5 |
| 8 | 5.16e-2 | 6.36e-4 |

This does not flatten the growth with bit width, it lowers it by one to two orders
of magnitude. The error remains proportional to the layer output while the
residual shrinks, so it is a remedy with a working range rather than a general
solution. Above 8 bits, or on much wider layers, re-measure before assuming it
still holds.

Whether to pay three times the arithmetic is a Phase 2 decision with a stated
criterion in MEMORY.md D-016, not a default. The accuracy question is settled; the
quality-per-second question is not.

### 5.10 The searched scale grid

Section 5.2 says the v2 search produces `s_init` per group; this section fixes
the search itself, which the published method leaves open and which the
reference implementation decides in code. MRound follows the reference's
choices, candidate for candidate, so that the two searched initializations
start every layer from the same grid (MEMORY.md D-048).

```
    nmax          = 2^(b - 1)
    anchor        = the element of largest magnitude in the group, at the
                    first index on an exact tie, mapped onto -nmax
    candidate i   effective range  r_i = nmax - step * i
                  inverse scale    1/s_i = -r_i / anchor
                  codes            clamp(round_half_to_even(w / s_i), -nmax, nmax - 1)
                  loss             sum over the group of qw * (s_i * codes - w)^2
```

- At 2 bits the window is fixed: `i` runs over `[-90, 90]` excluding zero at
  `step = 0.01`, so 180 candidates spanning effective ranges 1.10 to 2.90
  around `nmax = 2`. The ratio below does not apply.
- At every other width there are 200 candidates: `step = nmax * ratio / 100`
  and `i` runs over `[-100, 100]` excluding zero, with `ratio = 0.75`. The
  reference exposes the ratio through an environment variable; MRound exposes
  it as `SearchGrid.ratio` and changes nothing else.
- The anchor's own scale (`i = 0`) is evaluated first and a candidate replaces
  it only on strict improvement; among candidates, iteration runs from the
  widest effective range to the narrowest and ties keep the earlier one.
- `qw` is the per input channel importance where the search is importance
  weighted, and 1 otherwise.
- An all zero group returns scale 0, which no candidate can beat.

Two conventions are worth naming because they look like inconsistencies and
are not. First, the anchor's tie rule (first index of the largest magnitude)
differs from the observed range scale's rule in section 5.2, where the positive
extreme wins a tie; each mirrors the reference code path it reimplements, and
the two agree everywhere except on exact ties, which have measure zero on
float32 weights. Second, a negative `i` widens the effective range, so the
largest elements land beyond the clamp: it clips the tail and buys resolution
for the bulk. A positive `i` narrows it, so the largest element lands inside
the code range and the outer codes go unused: coarser everywhere, never
clipped. The module docstring once described this backwards and was corrected
in the audit of 2026-09-23.

---

## 6. Formats

### 6.1 Packing

Integer codes pack into unsigned 32-bit words along the input dimension, with
element zero in the least significant field.

When the bit width divides 32 evenly (2, 4, 8), each word holds `32/b` elements
at fixed offsets. Otherwise (3, 5, 6, 7), elements are laid out as a contiguous
little-endian bitstream in which element `i` occupies absolute bits `[i*b,
(i+1)*b)` and elements may straddle word boundaries; 32 elements occupy exactly
`b` words.

Packing correctness is verified by exhaustive round-trip tests over random data
at every supported bit width. This is cheap to test and expensive to get wrong
silently.

### 6.2 MLX-native checkpoints

MLX quantized layers reconstruct weights as

```
    W = scales * q_u + biases
```

where `q_u` is the unsigned stored code. Both of MRound's modes reduce to this
form, but they arrive by different routes, because section 5.2's `q` is a signed
code in the symmetric branch and already an unsigned code in the asymmetric one.

**Symmetric.** Section 5.2 produces `W = s * q` with `q` signed in
`[-2^(b-1), 2^(b-1) - 1]`. Storage requires an unsigned code, so a fixed integer
offset `z = 2^(b-1)` is applied at pack time, `q_u = q + z`, which maps the range
exactly:

```
    [ -2^(b-1),  2^(b-1) - 1 ]  +  2^(b-1)  =  [ 0,  2^b - 1 ]
```

with no clipping and no wasted codes. Then `W = s * (q_u - z)`.

**Asymmetric.** Section 5.2 already produces `W = s * (q - z)` with `q` unsigned
in `[0, 2^b - 1]`, so `q_u = q` and the learned `z` is used directly.

Both cases are now `W = s * (q_u - z) = s*q_u - s*z`, giving

```
    scales = s
    biases = -s * z
```

**Scales may be negative.** In symmetric mode `s` carries the sign opposite to
the group's larger-magnitude extreme (section 5.2.1), so exported scales are signed and the
corresponding biases flip sign with them. The affine reconstruction is correct
either way, since nothing in `scales * q_u + biases` assumes positivity. Any
consuming format that requires non-negative scales must be handled explicitly by
its exporter, either by negating the codes within the group or by refusing the
group; silently taking an absolute value would corrupt the weights.

Per-layer bit widths and group sizes are recorded in the checkpoint
configuration so that mixed-precision models load correctly.

### 6.3 GGUF

*(Planned, ROADMAP.md Phase 4. `formats/gguf_export.py` raises
`NotImplementedError` and `api.quantize` refuses `export_format="gguf"`; this
section specifies the exporter that does not exist yet.)*

GGUF export maps MRound schemes onto GGUF quantization types where a faithful
mapping exists. Where none exists, export fails with a clear message rather than
silently approximating. A quantizer that quietly produces something other than
what was requested is worse than one that refuses.

The signed-scale case from section 6.2 is the first place this rule bites: GGUF's
block formats assume a particular scale convention, and a group whose scale is
negative must be normalized into it, not coerced. Normalization by negating the
codes within the group is exact and is the preferred handling.

---

## 7. MLX implementation hazards

These are the differences that actually bite. They are listed with their
resolutions so that the resolutions are adopted from the first commit rather
than retrofitted after an architecture has calcified around the wrong
assumption.

**Lazy evaluation.** MLX builds a computation graph and evaluates it when a
result is needed. Two failure modes follow. Reading a scalar every iteration, as
a naive loss log would, forces a synchronous evaluation each step and can
dominate runtime. Never evaluating lets the graph grow without bound and
exhausts memory. Resolution: evaluation boundaries are explicit design decisions,
placed once per optimization step at a defined point, with loss values
accumulated on device and read only periodically.

**Immutable arrays.** There is no in-place mutation. Any pattern that modifies a
parameter during the forward pass, such as clamping it mid-computation, has no
translation. Resolution: parameters are projected into their valid ranges as an
explicit step after the optimizer update, never inside the traced function.

**Functional optimizer state.** MLX optimizers operate over parameter trees
rather than mutating objects identified by reference. Resolution: MRound's
optimizer follows the MLX convention natively. Since signed gradient descent is
one line of arithmetic, this is not a porting problem, it is simply how it is
written.

**No forward hooks.** MLX has no mechanism to attach a callback to a module's
forward pass. Importance matrix accumulation and activation capture therefore
require explicit instrumentation. Resolution: the block runner runs the model to
a boundary explicitly rather than relying on interception. This is more code but
it is also more legible.

**No ambient gradient mode.** There is no global flag to disable gradient
tracking. Gradients are produced by a function transformation, so code either is
or is not inside the transformed function. Resolution: the traced region is kept
small and explicit, which is good practice regardless.

**Limited fancy indexing.** Boolean mask assignment is not available in the form
eager frameworks provide. Resolution: the outlier-suppression mask is built by a
scatter or by a threshold comparison rather than by assignment into a boolean
array.

**No float64.** Metal has no double precision. Resolution: the dynamic
programming allocator, the only place where wide accumulation might be wanted,
runs on the CPU where float64 is available.

**Custom kernels.** MLX exposes custom Metal kernels through a source-string
interface that generates the signature and handles dispatch. These are available
when needed. Their differentiability must be verified before one is used inside
the tuning loop rather than assumed; the safe pattern is to use custom kernels
for non-differentiated stages, such as the scale search and packing, and to keep
the gradient path in standard MLX operations.

---

## 8. Validation methodology

Four levels, cheapest first. The first runs in continuous integration; the
state of the other three is stated beside each, because two of them are
planned rather than built.

**Unit tests.** Synthetic data, no model. Every mathematical component tested
against a hand-computed or analytically-known result. Round-trip properties
where they exist: pack then unpack is the identity, and the packed layout is
word identical with `mx.quantize` at every width and group size MLX accepts.
The allocator tested against exhaustive search on small instances, including
cases constructed so that greedy allocation is provably wrong. The NumPy
reference's gradients are checked against finite differences of the
straight-through surrogate on every branch, the searched one included, and the
MLX core against the reference function for function (`tests/unit/`, about
850 tests).

**Parity tests.** *(Planned.)* Against a dumped reference corpus, at the
granularity of a single layer, with byte-identical inputs: scales, integer
codes, the fraction of rounding decisions that differ, and layer output error.
`tests/parity/` holds only its README today; the layer level corpus and its
generator have not been written. What exists instead is a model level
comparison harness, which quantizes one model with both implementations under
matched settings and scores both with one evaluator (MEMORY.md D-037), and
`scripts/check_mlx.sh`. Seed-matched
trajectory comparison across frameworks is explicitly not attempted, because
it is not achievable and pursuing it wastes effort.

**Integration tests.** *(Planned.)* End to end on models small enough to run
in continuous integration. `tests/integration/` is empty; the end to end path
is exercised by `examples/tune_model.py` and the parity harness on Apple
hardware.

**Quality benchmarks.** Perplexity and, once `eval/zeroshot.py` exists,
zero-shot task accuracy on real models. Too slow for continuous integration;
run by hand on Apple hardware and on the Linux reference machine, and every
run that produces a quotable number appends one line to
`benchmarks/ledger.jsonl` the day it is produced (MEMORY.md D-029), citing a
hashed environment manifest (D-038). The ledger does not yet record the
commit that produced a line; adding it is an open item.

### 8.1 The perplexity evaluator

The conventions that fix every quoted perplexity live in `eval/perplexity.py`
and are stated here because D-037 requires both sides of a comparison to be
scored by the same evaluator. Logits are cast to float32 before the loss. Text
is scored in windows of `seq_len` tokens (2048 by default) with a stride equal
to the window unless a shorter one is asked for, so windows do not overlap by
default; the first token of each window has no predecessor inside its window
and is not scored, which is why a request for 65536 tokens scores
`32 x 2047 = 65504` targets. Under an overlapping stride only the first
`stride` tokens of each window after the first are scored, so no target is
counted twice. The result is `exp` of the mean negative log likelihood over
the scored targets, and the ledger's `evaluation.tokens` is that scored count.
The parity harness scores every checkpoint at float32 weights; the example
harness scores each checkpoint at its own dtype, and since 2026-09-17 both
record the dtype beside the count.

---

## 9. The Neural Engine, honestly

The Neural Engine is a fixed-function accelerator for neural network inference.
It is efficient at what it does and it is not a general compute device.

**What it cannot do.** It cannot run MRound's tuning loop. It has no automatic
differentiation, no custom kernels, and no mechanism for executing arbitrary
programs. It is reachable only by compiling a model through Core ML, and the
compiler decides what runs where. There is no path by which signed gradient
descent over rounding parameters executes on the ANE, and no amount of
engineering changes this.

**What it might do.** Run a quantized model for inference, potentially at lower
energy per token than the GPU. Whether that translates into a latency advantage
for autoregressive generation on large models is an empirical question this
project treats as open.

**The constraints that shape any attempt.** Core ML's low-bit weight support is
principally weight compression: weights are stored compressed and, in most
configurations, decompressed before the arithmetic, so the benefit is memory
footprint and bandwidth rather than arithmetic throughput. Per-block scale
granularity, which is what MRound produces and what gives group-wise
quantization its quality, is better suited to GPU execution; per-channel scales
are preferred for ANE execution, which is a coarser granularity and costs
quality. Integer 8-bit appears to be the best-supported low-precision path. The
2-bit work that is central to MRound's purpose is the least likely to map well.

Every statement in the preceding paragraph comes from Apple's published
documentation rather than from measurement, which is precisely why Phase 7 is
scoped to establish them empirically. None of them should be repeated elsewhere
in this project's documentation as established fact until that happens.

**Therefore.** The ANE is a Phase 7 research track with a timebox and permission
to conclude that it is not worthwhile. MRound will publish measurements, not
aspirations, and a negative result will be reported as clearly as a positive
one. Any claim of ANE acceleration in this project's documentation must be
accompanied by a benchmark or removed.

---

## 10. Glossary

**Block-wise quantization.** Quantizing one transformer block at a time,
optimizing each to reproduce its own output. Keeps memory bounded and avoids a
full-model backward pass.

**Calibration data.** A small unlabeled sample of representative text used to
produce the activations against which reconstruction error is measured.

**Group size.** The number of consecutive weights sharing one scale. Smaller
groups give better quality and more metadata overhead. 128 is the common
default in the literature and `QuantScheme`'s; every measured MRound run uses
64 (section 5.4).

**Importance matrix.** Accumulated sums of squared activations per input
channel, used to weight reconstruction error by how much each input actually
matters.

**Mixed precision.** Different bit widths for different layers, allocated by
measured sensitivity under a total size budget.

**Straight-through estimator.** A rounding operation whose forward pass rounds
and whose backward pass passes gradients through unchanged, making the
non-differentiable differentiable enough to optimize.

**Signed gradient descent.** An update rule using only the sign of the gradient,
so every parameter moves by exactly the learning rate each step.

**Weight-only quantization.** Compressing weights while leaving activations in
floating point. Simpler and more robust than also quantizing activations, and
sufficient when memory bandwidth is the binding constraint, as it is for
single-stream generation.
