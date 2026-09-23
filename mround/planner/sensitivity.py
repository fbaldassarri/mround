# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Layer sensitivity scoring.

Estimates how much each layer's quantization contributes to the overall loss
increase, so that :mod:`mround.planner.allocator` can spend bits where they buy
the most.

The score combines gradient magnitude with the perturbation quantization
actually introduces::

    DeltaLoss = || g * (W_q - W_f) ||_1

Note what this is and is not. The first-order term is ``g . dW``, a signed sum
in which opposing perturbations cancel. Taking absolute values before summing
gives an upper bound on its magnitude rather than the term itself, so this is
better described as a cancellation-free proxy. That distinction matters: a
signed sum can report a near-zero score for a layer being damaged badly in both
directions.

Two functions, and a deliberate boundary between them. :func:`delta_loss` is
the executable definition of the score, in NumPy, and doubles as the oracle the
MLX reduction in :mod:`mround.pipeline.scoring` is tested against, the same
relationship ``reference/`` has to ``core/``. :func:`score_layer_options`
consumes only scalars: the pipeline reduces gradients against perturbations on
the device and hands one float per layer per width across this boundary, so
model-sized arrays never enter the planner. That is also why this module's
signature changed from the stub it replaced, which imagined the planner
receiving whole gradient dictionaries.

Where the gradients come from is the pipeline's business, specified in
DOCUMENTATION.md section 1.5: by default one causal-LM backward per batch at
the widest candidate width, scoring every width's perturbation against it
(MEMORY.md D-035), with the reference implementation's procedure, one backward
per width on the model round-to-nearest quantized at that width, kept as the
``own_width`` control. Scores from different calibration passes are not
comparable, and nothing here can detect that mistake.

Specification: DOCUMENTATION.md section 1.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mround.planner.allocator import LayerOption

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

    from mround.schemes import QuantScheme

__all__ = [
    "SENSITIVITY_BATCH_TOKENS",
    "SENSITIVITY_SAMPLES",
    "SENSITIVITY_SEQ_LEN",
    "delta_loss",
    "groups_per_row",
    "mixed_budget_bits",
    "score_layer_options",
    "scoring_batch_size",
    "storage_bits",
]

# Calibration used for sensitivity estimation, which is far cheaper than tuning
# and needs correspondingly less data. DOCUMENTATION.md section 1.5.
#
# These were 16 sequences of 256 tokens, the reference implementation's own
# defaults, adopted for comparability rather than derived. They are now the
# budget every measurement in this repository was actually taken at, which is
# not the same thing and is the reason they changed. Same model, same seed, same
# reference checkpoint: 24.2539 at 16x256 against 23.1140 at 128x1024, or 4.70
# percent, and not one of the fifty one ledger rows was produced at the old
# values. What made the cheap budget defensible was its cost, and D-042's token
# constant batch removed that: at a fixed batch of 8 the scoring pass was 1512
# seconds, 41 percent of a run, while every run since has measured it at 323 to
# 388 seconds on Qwen2.5-0.5B and 119 on SmolLM2, 6 to 10 percent, with the peak
# unchanged because the batch holds the token count still. MEMORY.md D-031.
SENSITIVITY_SAMPLES: int = 128
SENSITIVITY_SEQ_LEN: int = 1024

# Tokens per scoring batch, which is what actually sets the activation peak of
# a scoring pass. 2048 is unchanged by the budget above and was chosen to be:
# it is what the old defaults produced at the batch size of 8 that was hard
# coded before them, and what the new ones produce at a batch of 2, so the
# activation peak of a scoring pass is the same whichever budget it runs.
#
# It becomes a constant rather than a batch count because the scoring budget is
# now an argument (the reference asks for 128 sequences of 1024 tokens on
# 2-bit schemes). A fixed batch of 8 holds the batch count still and lets the
# activations grow with the sequence, and the dominant activation in a causal
# LM backward is the logits, batch times tokens times a vocabulary of 151936 on
# Qwen2.5: four times the sequence at a fixed batch is four times that tensor
# and everything the backward keeps alongside it. Measured on log0059: the
# 0.5B mixed run peaked at 17.14 GB at 8 by 256, and the same run at 8 by 1024
# stopped making visible progress on a 32 GB machine.
SENSITIVITY_BATCH_TOKENS: int = 2048


def scoring_batch_size(
    n_samples: int,
    seq_len: int,
    *,
    budget_tokens: int = SENSITIVITY_BATCH_TOKENS,
) -> int:
    """Sequences per scoring batch, at a fixed token count per batch.

    Splitting the same sequences into more, smaller batches does not change
    what the allocator decides. Each batch contributes its own
    ``sum |gradient * perturbation|`` to a running total, so a finer split
    raises every layer's score at every width by about the same factor, and the
    allocator minimizes total predicted loss under a budget, an objective whose
    argmin is invariant to a positive scale. What does change is the recorded
    ``predicted_loss``, which is therefore comparable only within one batching.

    Args:
        n_samples: Sequences in the scoring draw.
        seq_len: Tokens per sequence.
        budget_tokens: Tokens one batch may hold.

    Returns:
        At least one sequence, at most the whole draw.
    """
    per_batch = max(1, int(budget_tokens) // max(1, int(seq_len)))
    return max(1, min(int(n_samples), per_batch))


def delta_loss(
    weight: npt.ArrayLike,
    quantized_weight: npt.ArrayLike,
    gradient: npt.ArrayLike,
) -> float:
    """Score one layer at one candidate width: the L1 gradient-perturbation product.

    Absolute values are taken elementwise before the sum, never after, so
    opposing perturbations cannot cancel. Accumulation is float64 on the CPU,
    which costs nothing at this call rate and removes summation order from the
    result.

    Args:
        weight: Full-precision weights.
        quantized_weight: The same weights after quantize-dequantize at the
            candidate width.
        gradient: Loss gradient with respect to the quantize-dequantized
            weight, from a calibration backward pass.

    Returns:
        The scalar score. Comparable across layers only when the gradients come
        from the same calibration pass.

    Raises:
        ValueError: If the three arrays do not share one shape.
    """
    w = np.asarray(weight, dtype=np.float64)
    q = np.asarray(quantized_weight, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64)
    if not (w.shape == q.shape == g.shape):
        msg = (
            f"weight {w.shape}, quantized weight {q.shape}, and gradient "
            f"{g.shape} must share one shape"
        )
        raise ValueError(msg)
    return float(np.sum(np.abs(g * (q - w))))


def storage_bits(
    shape: tuple[int, int],
    bits: int,
    group_size: int,
    *,
    dtype_bits: int = 16,
) -> int:
    """Total bits one layer costs at one width, counting the scale metadata.

    The same accounting the export path reports (MEMORY.md D-019): packed codes
    plus one scale and one bias per group at the checkpoint's dtype. The
    metadata is half a bit per weight at group size 64 and 16-bit scales, which
    is not negligible against a 2-bit code, and omitting it would bias the
    allocator toward small group sizes the budget cannot actually afford.

    Args:
        shape: ``(out_features, in_features)``.
        bits: Candidate width.
        group_size: Elements per quantization group along the input axis.
        dtype_bits: Width of one stored scale or bias, 16 for the float16 and
            bfloat16 checkpoints MRound writes.

    Returns:
        The layer's storage in bits.

    Raises:
        ValueError: If the shape is not positive or does not divide into whole
            groups. A layer that cannot be grouped stays dense and must not be
            costed as if it could be quantized.
    """
    out_features, in_features = shape
    if out_features <= 0 or in_features <= 0:
        msg = f"shape must be positive, got {shape}"
        raise ValueError(msg)
    groups = out_features * groups_per_row(in_features, group_size)
    return out_features * in_features * bits + 2 * groups * dtype_bits


def groups_per_row(in_features: int, group_size: int) -> int:
    """Scales one output channel needs, with ``-1`` meaning one per channel.

    The same rule as :meth:`mround.schemes.QuantScheme.groups_per_row`, kept
    here in integer form because the cost model is called with bare shapes.
    Without the per channel case ``in_features % -1`` is zero and
    ``in_features // -1`` is negative, which used to cost a per channel layer
    a negative amount of metadata and refuse the allocation with a message
    about a negative cost.

    Raises:
        ValueError: If the row does not divide into whole groups. A layer that
            cannot be grouped stays dense and must not be costed as if it
            could be quantized.
    """
    if group_size == -1:
        return 1
    if group_size <= 0 or in_features % group_size:
        msg = (
            f"in_features {in_features} is not a multiple of group size "
            f"{group_size}; this layer stays dense and has no quantized cost"
        )
        raise ValueError(msg)
    return in_features // group_size


def mixed_budget_bits(
    shapes: Mapping[str, tuple[int, int]],
    average_bits: float,
    group_size: int,
    *,
    dtype_bits: int = 16,
) -> int:
    """The size budget an average code width implies over these layers.

    Code bits are budgeted as the layers' total element count times the
    average, and the scale metadata is added on top at its exact per-layer
    cost, because metadata does not vary with the chosen width at a fixed
    group size. The property that makes this the right definition, pinned by a
    test: at an integer average the budget equals the exact storage of the
    uniform allocation at that width, so "mixed at an average of n" and
    "uniform n" compete under the same ceiling and a quality difference
    between them is attributable to the allocation rather than to a budget
    quietly favoring one side.

    Args:
        shapes: Layer name to ``(out_features, in_features)`` for every layer
            the allocator will choose over.
        average_bits: Target average code bits per weight across those layers.
        group_size: Elements per quantization group along the input axis.
        dtype_bits: Width of one stored scale or bias.

    Returns:
        The budget in bits, for :func:`mround.planner.allocator.allocate_bits`.

    Raises:
        ValueError: If there are no layers, the average is not positive, or a
            shape cannot be grouped.
    """
    if not shapes:
        msg = "no layers to budget over"
        raise ValueError(msg)
    if average_bits <= 0:
        msg = f"average_bits must be positive, got {average_bits}"
        raise ValueError(msg)
    elements = 0
    metadata = 0
    for name, (out_features, in_features) in shapes.items():
        if out_features <= 0 or in_features <= 0:
            msg = f"layer {name!r} with shape {(out_features, in_features)} cannot be grouped"
            raise ValueError(msg)
        try:
            groups = groups_per_row(in_features, group_size)
        except ValueError as exc:
            msg = f"layer {name!r} with shape {(out_features, in_features)} cannot be grouped"
            raise ValueError(msg) from exc
        elements += out_features * in_features
        metadata += 2 * out_features * groups * dtype_bits
    return int(elements * average_bits) + metadata


def score_layer_options(
    scores: Mapping[str, Mapping[int, float]],
    shapes: Mapping[str, tuple[int, int]],
    base_scheme: QuantScheme,
    *,
    dtype_bits: int = 16,
) -> dict[str, list[LayerOption]]:
    """Turn per-layer per-width scores into the allocator's input.

    Produces exactly what :func:`mround.planner.allocator.allocate_bits`
    consumes, which is the only reason the two modules need to agree on
    anything. Every layer must be scored at the same set of widths: a width
    missing for one layer means its scoring pass did not cover it, and filling
    the gap with anything would hand the allocator an invented number.

    Args:
        scores: Layer name to a mapping of candidate width to its DeltaLoss,
            all from the same calibration data.
        shapes: Layer name to ``(out_features, in_features)``, for the storage
            cost.
        base_scheme: Scheme whose group size applies to every option; only the
            bit width varies.
        dtype_bits: Width of one stored scale or bias.

    Returns:
        Layer name to its scored options, widths ascending, insertion order
        following ``scores``.

    Raises:
        ValueError: If ``scores`` and ``shapes`` do not cover the same layers,
            the layers do not share one width set, a score is not finite and
            non-negative, or a shape cannot be grouped.
    """
    if set(scores) != set(shapes):
        only_scores = sorted(set(scores) - set(shapes))
        only_shapes = sorted(set(shapes) - set(scores))
        msg = (
            f"scores and shapes must cover the same layers; "
            f"only scored: {only_scores}, only shaped: {only_shapes}"
        )
        raise ValueError(msg)
    if not scores:
        msg = "no layers to score"
        raise ValueError(msg)

    widths: set[int] | None = None
    for name, per_width in scores.items():
        if widths is None:
            widths = set(per_width)
            if not widths:
                msg = f"layer {name!r} has no scored widths"
                raise ValueError(msg)
        elif set(per_width) != widths:
            msg = (
                f"layer {name!r} is scored at widths {sorted(per_width)} but "
                f"others at {sorted(widths)}; every layer must come from the "
                f"same scoring passes"
            )
            raise ValueError(msg)

    options: dict[str, list[LayerOption]] = {}
    for name, per_width in scores.items():
        candidates: list[LayerOption] = []
        for width in sorted(per_width):
            score = per_width[width]
            if not np.isfinite(score) or score < 0.0:
                msg = f"layer {name!r} at {width} bits has invalid score {score!r}"
                raise ValueError(msg)
            out_features, in_features = shapes[name]
            candidates.append(
                LayerOption(
                    bits=width,
                    cost_bits=storage_bits(
                        shapes[name], width, base_scheme.group_size, dtype_bits=dtype_bits
                    ),
                    delta_loss=score,
                    elements=out_features * in_features,
                )
            )
        options[name] = candidates
    return options
