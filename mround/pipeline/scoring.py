# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Whole-model sensitivity scoring for mixed precision.

Produces the per-layer, per-width DeltaLoss scores the planner allocates bits
from: one causal-LM backward per candidate width, on the model round-to-nearest
quantized at that width, accumulating ``sum |gradient * perturbation|`` per
layer. The procedure is the reference implementation's (analysis 08): gradients
are taken at the quantized model rather than the original, each candidate gets
its own backward, and the quantizer during scoring is plain round-to-nearest
with nothing learned. For MRound's weight-only schemes the activation term the
reference conditionally adds is identically absent, so the score is the weight
term alone.

One deliberate divergence, recorded in MEMORY.md D-031: when the scheme says
``SEARCHED``, the quantize-dequantize here uses the searched grid, because the
score exists to predict the damage the actual quantizer will do and MRound's
2-bit quantizer is the searched one. The reference has no searched grid at
scoring time to mirror. The search runs without importance, since no activation
statistics exist before calibration ever flows, which is the same policy the
remainder tensors already follow (D-030).

A second divergence is measured rather than structural (MEMORY.md D-034): the
API's default ``gradient_source`` is ``"widest"``, one backward at the widest
candidate scoring every width's perturbation, because gradients taken through
a network the lowest width has destroyed starve the late blocks, and the A/B
showed it on both averages tested. This module's own default stays
``"own_width"``, the reference-faithful primitive, so the reference procedure
remains one call away and the oracle tests keep their meaning.

MLX has no per-tensor backward hooks, so where the reference grabs ``dL/dW_q``
with a gradient hook on the quantize-dequantize output, this module makes the
quantized weights explicit function inputs and asks ``mx.grad`` for their
cotangents directly, which is the same quantity by construction: the gradient
at that node does not care how the values in it were produced.

The reduction ``sum |g * (W_q - W)|`` runs on the device in float32, one scalar
per layer per batch, so model-sized arrays never cross into the planner. Its
NumPy definition, :func:`mround.planner.sensitivity.delta_loss`, is the oracle
the MLX reduction is tested against.

**Nothing here is silent.** Scoring prints one line per batch, with elapsed
time, a projection, and the peak allocation. The reason is log0059: a scoring
budget thirty two times the default turned a two minute pass into a two hour
one with no output at all between the model download and the first block, and
the only honest reading from outside was that the run had hung. A pass this
long that says nothing is indistinguishable from a broken one, and the peak
allocation in the same line is what separates working from swapping.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx import nn

from mround.core.quantizer import fake_quantize
from mround.core.scale_search import search_scales
from mround.pipeline.blocks import discover_blocks, iter_quantizable_linears
from mround.pipeline.device import peak_memory
from mround.schemes import DEFAULT_EPS_FP32, QuantScheme, ScaleInit

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["ScoringProgress", "SensitivityScores", "score_widths"]


@dataclasses.dataclass(slots=True)
class SensitivityScores:
    """What one scoring run measured, ready for the planner.

    Attributes:
        scores: Layer path to a mapping of candidate width to its DeltaLoss,
            exactly the shape
            :func:`mround.planner.sensitivity.score_layer_options` consumes.
        shapes: Layer path to ``(out_features, in_features)``, for the storage
            cost model.
        skipped: Layers no candidate can quantize, with the reason, so a layer
            absent from the allocation is absent by report rather than by
            silence.
        seconds: Wall clock the whole scoring took, for the ledger.
    """

    scores: dict[str, dict[int, float]]
    shapes: dict[str, tuple[int, int]]
    skipped: tuple[tuple[str, str], ...]
    seconds: float


@dataclasses.dataclass(frozen=True, slots=True)
class ScoringProgress:
    """One reportable moment inside a scoring pass.

    Attributes:
        stage: ``"perturbations"`` for the up-front quantize-dequantize and
            difference construction, ``"backward"`` for one batch of the
            gradient loop.
        index: How many units of this stage are finished.
        total: How many there are. For the backward loop this counts batches
            once per gradient pass, so it is the batch count under ``widest``
            and the batch count times the candidate count under ``own_width``.
        seconds: Wall clock since scoring started.
        peak_memory_bytes: Peak MLX allocation so far, zero when the build will
            not report one.
        stage_seconds: Wall clock since this stage started. Under ``widest``
            the perturbation stage precedes the first backward and can
            dominate the pass (43.6 of 50.6 seconds in ledger row 18), so a
            projection of the backward loop has to start its clock where the
            loop starts, or its first lines read several times too high and
            fall on every line after, which looks like a run speeding up.
    """

    stage: str
    index: int
    total: int
    seconds: float
    peak_memory_bytes: int
    stage_seconds: float = 0.0

    @property
    def projected_seconds(self) -> float:
        """Linear projection of the whole stage from what it has done so far."""
        if self.index < 1:
            return 0.0
        return self.stage_seconds / self.index * self.total

    @property
    def remaining_seconds(self) -> float:
        """What the projection says is still to come in this stage."""
        return max(0.0, self.projected_seconds - self.stage_seconds)

    def describe(self) -> str:
        """One line for a progress log."""
        peak = (
            ""
            if not self.peak_memory_bytes
            else f", peak memory {self.peak_memory_bytes / 1e9:.2f} GB"
        )
        if self.stage == "perturbations":
            return (
                f"  scoring: perturbations for {self.total} widths ready "
                f"in {self.seconds:.1f}s{peak}"
            )
        return (
            f"  scoring: backward {self.index}/{self.total} at {self.seconds:.0f}s, "
            f"about {self.remaining_seconds:.0f}s remaining{peak}"
        )


def _causal_loss(model: nn.Module, batch: mx.array) -> mx.array:
    """The calibration objective: mean shifted cross-entropy, float32.

    The same convention the perplexity harness scores with, and the reference
    scores against (labels are the inputs themselves). Mean reduction, so the
    gradient scale is independent of batch shape; the score itself remains
    extensive because it sums over batches and elements.
    """
    logits = model(batch[:, :-1]).astype(mx.float32)
    return mx.mean(nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none"))


def _quantized_weights(
    weights: dict[str, mx.array],
    scheme: QuantScheme,
    eps: float,
) -> dict[str, mx.array]:
    """Round-to-nearest quantize-dequantize of every eligible weight.

    ``params=None`` is the quantizer's round-to-nearest path: nothing learned,
    scalar unit coefficients. Under a searched scheme the grid is searched
    first, unweighted, and handed in as the initial scale.
    """
    qdq: dict[str, mx.array] = {}
    for path, weight in weights.items():
        init_scale = None
        if scheme.scale_init is ScaleInit.SEARCHED:
            init_scale = search_scales(weight, scheme)
        qdq[path] = fake_quantize(weight, None, scheme, eps=eps, init_scale=init_scale)
    return qdq


def _scores_from_own_width(
    grad_fn: Any,
    batches: Sequence[mx.array],
    weights32: dict[str, mx.array],
    *,
    schemes: dict[int, QuantScheme],
    eps: float,
    paths: list[str],
    report: Callable[[str, int, int], None],
) -> dict[str, dict[int, float]]:
    """The reference's procedure: each width's backward at its own model.

    One width is resident at a time, so this holds three model-sized float32
    copies (the originals, one quantize-dequantize, one perturbation) however
    many candidates there are. See MEMORY.md D-036.
    """
    scores: dict[str, dict[int, float]] = {path: {} for path in paths}
    passes = len(batches) * len(schemes)
    done = 0
    for width in sorted(schemes):
        qdq = _quantized_weights(weights32, schemes[width], eps)
        diff = {path: qdq[path] - weights32[path] for path in paths}
        mx.eval(list(qdq.values()), list(diff.values()))

        accumulated = {path: mx.zeros((), dtype=mx.float32) for path in paths}
        for batch in batches:
            gradients = grad_fn(qdq, batch)
            for path in paths:
                term = mx.sum(mx.abs(gradients[path] * diff[path]))
                accumulated[path] = accumulated[path] + term
            # One synchronization per batch bounds the graph; there are few
            # batches and each holds a whole-model backward.
            mx.eval(list(accumulated.values()))
            done += 1
            report("backward", done, passes)
        for path in paths:
            scores[path][width] = float(accumulated[path])
    return scores


def _scores_from_widest(
    grad_fn: Any,
    batches: Sequence[mx.array],
    weights32: dict[str, mx.array],
    *,
    schemes: dict[int, QuantScheme],
    eps: float,
    paths: list[str],
    report: Callable[[str, int, int], None],
) -> dict[str, dict[int, float]]:
    """One backward per batch at the widest candidate, every width scored on it.

    The gradient needs only the widest width's quantize-dequantize; every
    other width contributes nothing but its perturbation, so each narrower
    quantized copy is released as soon as its difference exists. ``weights32``
    is consumed rather than borrowed: it is cleared once the last perturbation
    is formed, because holding the float32 originals through the batch loop
    costs a whole model copy for nothing. The batch loop therefore holds one
    quantized copy plus one perturbation per candidate, which is the floor for
    this schedule: the gradient is nonlinear in the perturbation, so the
    perturbations cannot be folded together and a width dropped now would cost
    a second backward later. See MEMORY.md D-036.
    """
    widest = max(schemes)
    anchor = _quantized_weights(weights32, schemes[widest], eps)
    mx.eval(list(anchor.values()))

    diffs: dict[int, dict[str, mx.array]] = {}
    for width in sorted(schemes):
        qdq = anchor if width == widest else _quantized_weights(weights32, schemes[width], eps)
        diffs[width] = {path: qdq[path] - weights32[path] for path in paths}
        mx.eval(list(diffs[width].values()))
        del qdq
    weights32.clear()
    report("perturbations", len(schemes), len(schemes))

    accumulated = {
        width: {path: mx.zeros((), dtype=mx.float32) for path in paths} for width in diffs
    }
    for index, batch in enumerate(batches, start=1):
        gradients = grad_fn(anchor, batch)
        for width in diffs:
            for path in paths:
                term = mx.sum(mx.abs(gradients[path] * diffs[width][path]))
                accumulated[width][path] = accumulated[width][path] + term
        mx.eval([value for per in accumulated.values() for value in per.values()])
        report("backward", index, len(batches))
    return {path: {width: float(accumulated[width][path]) for width in diffs} for path in paths}


def score_widths(
    model: nn.Module,
    batches: Sequence[mx.array],
    widths: Sequence[int],
    base_scheme: QuantScheme,
    *,
    expected_blocks: int | None = None,
    eps: float = DEFAULT_EPS_FP32,
    gradient_source: str = "own_width",
    progress: Callable[[ScoringProgress], None] | None = None,
) -> SensitivityScores:
    """Score every block layer at every candidate width.

    The layers scored are exactly the layers the block loop tunes, keyed by the
    same full dotted paths, so the allocation that comes back plugs into the
    runner's ``scheme_by_layer`` without translation. Embeddings and anything
    else outside the blocks never enter: the reference excludes embeddings from
    scoring because the tiny calibration reaches too few rows of a lookup for
    the score to mean anything, and what MRound does with the remainder tensors
    is the wiring's decision, not this module's.

    Args:
        model: A loaded full-precision MLX model. Left exactly as found: the
            weights are swapped for quantized ones during each backward and
            restored before returning, on every path including failure.
        batches: Tokenized calibration batches, ``(batch, tokens)`` int arrays.
            The scoring default is far smaller than tuning calibration; see
            ``SENSITIVITY_SAMPLES`` in :mod:`mround.planner.sensitivity`.
        widths: Candidate bit widths.
        base_scheme: Scheme whose group size, symmetry, and scale
            initialization every candidate shares; only the width varies.
        expected_blocks: Block count from the model configuration, checked
            against what discovery finds.
        eps: Scale epsilon.
        gradient_source: Where the gradients that multiply each width's
            perturbation come from. ``"own_width"`` is the reference's
            procedure: each width gets its own backward on the model quantized
            at that width. ``"widest"`` takes one backward per batch at the
            widest candidate and scores every width's perturbation against it,
            targeting the mechanism the first measured allocation exposed
            (MEMORY.md D-033): gradients taken through a network the lowest
            width has largely destroyed discriminate poorly and starve the
            late blocks. At the widest width the two modes coincide exactly,
            gradients and perturbations both, which is the invariant the tests
            pin.
        progress: Called with a :class:`ScoringProgress` after the up-front
            perturbations (a stage only ``widest`` has) and after every
            gradient pass. Scoring at the budget
            the reference recommends for 2-bit schemes is an hours-long phase
            that produces no other output, so a caller with a console should
            pass something here.

    Returns:
        Scores, shapes, skipped layers, and wall clock.

    Raises:
        ValueError: If no widths are given, every layer is skipped, or
            ``gradient_source`` is unknown.
        ArchitectureError: If the block stack cannot be located.
    """
    if not widths:
        msg = "no candidate widths to score"
        raise ValueError(msg)
    if gradient_source not in ("own_width", "widest"):
        msg = f"gradient_source must be 'own_width' or 'widest', got {gradient_source!r}"
        raise ValueError(msg)
    started = time.perf_counter()

    blocks = discover_blocks(model, expected=expected_blocks)
    eligible: dict[str, nn.Module] = {}
    skipped: list[tuple[str, str]] = []
    for block in blocks:
        for name, linear in iter_quantizable_linears(block.module):
            path = f"{block.name}.{name}"
            if linear.weight.shape[-1] % base_scheme.group_size:
                skipped.append(
                    (
                        path,
                        f"last dimension {linear.weight.shape[-1]} is not a "
                        f"multiple of {base_scheme.group_size}",
                    )
                )
                continue
            eligible[path] = linear
    if not eligible:
        msg = "no layer in any block can be quantized at this group size"
        raise ValueError(msg)

    originals = {path: layer.weight for path, layer in eligible.items()}
    weights32 = {path: weight.astype(mx.float32) for path, weight in originals.items()}
    shapes = {
        path: (int(weight.shape[0]), int(weight.shape[1])) for path, weight in weights32.items()
    }

    def loss_from(qdq: dict[str, mx.array], batch: mx.array) -> mx.array:
        # The quantized weights enter as function inputs, so their cotangents
        # are exactly the gradients the reference takes at its qdq node.
        for path, layer in eligible.items():
            layer.weight = qdq[path]
        return _causal_loss(model, batch)

    grad_fn = mx.grad(loss_from)

    # How the candidates are held is a measured decision rather than a detail.
    # Materializing every width's quantize-dequantize and every width's
    # perturbation up front costs twice the candidate count in model-sized
    # float32 copies, which made scoring, not tuning, the peak of a mixed run
    # on a 0.5B model. The schedules below build what they need and release
    # the rest. See MEMORY.md D-036.
    schemes = {int(width): dataclasses.replace(base_scheme, bits=int(width)) for width in widths}

    current_stage = ""
    stage_started = started
    last_report = started

    def report(stage: str, index: int, total: int) -> None:
        nonlocal current_stage, stage_started, last_report
        if progress is None:
            return
        now = time.perf_counter()
        if stage != current_stage:
            # A stage starts when the previous one last reported: the first
            # backward report arrives only after a whole batch, so the loop's
            # clock has to begin at the end of the perturbation stage rather
            # than at its own first line.
            current_stage = stage
            stage_started = last_report
        last_report = now
        progress(
            ScoringProgress(
                stage=stage,
                index=index,
                total=total,
                seconds=now - started,
                peak_memory_bytes=peak_memory(),
                stage_seconds=now - stage_started,
            )
        )

    try:
        if gradient_source == "widest":
            scores = _scores_from_widest(
                grad_fn,
                batches,
                weights32,
                schemes=schemes,
                eps=eps,
                paths=list(eligible),
                report=report,
            )
        else:
            scores = _scores_from_own_width(
                grad_fn,
                batches,
                weights32,
                schemes=schemes,
                eps=eps,
                paths=list(eligible),
                report=report,
            )
    finally:
        for path, layer in eligible.items():
            layer.weight = originals[path]

    return SensitivityScores(
        scores=scores,
        shapes=shapes,
        skipped=tuple(skipped),
        seconds=time.perf_counter() - started,
    )
