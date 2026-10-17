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
budget thirty two times the former default turned a 200.7 second pass
(log0052) into a 1512.0 second one (row 38) with no output at all between the
model download and the first block, and the only honest reading from outside
was that the run had hung. A pass this long that says nothing is
indistinguishable from a broken one, and the peak allocation in the same line
is what separates working from swapping.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx import nn

from mround.core.quantizer import fake_quantize
from mround.core.scale_search import search_scales
from mround.pipeline.blocks import checkpointed_blocks, discover_blocks, iter_quantizable_linears
from mround.pipeline.device import peak_memory
from mround.schemes import DEFAULT_EPS_FP32, QuantScheme, ScaleInit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

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
            dominate the pass (193.9 of 329.7 seconds in log0061), so a
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


def _grids(originals: Mapping[str, mx.array], scheme: QuantScheme) -> dict[str, mx.array | None]:
    """Each layer's searched grid at one width, or ``None`` where it does not search.

    Searched once per width and kept, a few numbers per group, so that the
    width's quantize-dequantize can be rebuilt from the original weight
    whenever it is needed rather than held (MEMORY.md D-036's next lever,
    taken as M1c). Unweighted, since no activation statistics exist yet.
    """
    if scheme.scale_init is not ScaleInit.SEARCHED:
        return dict.fromkeys(originals)
    grids: dict[str, mx.array | None] = {}
    for path, weight in originals.items():
        grid = search_scales(weight.astype(mx.float32), scheme)
        mx.eval(grid)
        grids[path] = grid
    return grids


def _quantized(
    weight32: mx.array, scheme: QuantScheme, eps: float, grid: mx.array | None
) -> mx.array:
    """Round-to-nearest quantize-dequantize of one weight, nothing learned.

    ``params=None`` is the quantizer's round-to-nearest path, with scalar unit
    coefficients; under a searched scheme ``grid`` is the initial scale.
    """
    return fake_quantize(weight32, None, scheme, eps=eps, init_scale=grid)


def _anchor(
    originals: Mapping[str, mx.array],
    scheme: QuantScheme,
    eps: float,
    grids: Mapping[str, mx.array | None],
) -> dict[str, mx.array]:
    """The weights the backward runs at, built and evaluated one layer at a time."""
    anchor: dict[str, mx.array] = {}
    for path, weight in originals.items():
        anchor[path] = _quantized(weight.astype(mx.float32), scheme, eps, grids[path])
        mx.eval(anchor[path])
    return anchor


def _accumulate(
    gradients: dict[str, mx.array],
    by_block: Sequence[Sequence[str]],
    originals: Mapping[str, mx.array],
    perturbations: Mapping[int, Callable[[str, mx.array], mx.array]],
    accumulated: dict[int, dict[str, mx.array]],
) -> None:
    """Add one batch's ``sum |g * perturbation|`` to every layer's score at every width.

    Taken from the last block to the first, the order the backward produces
    the gradients in, and evaluated one layer and one width at a time, with
    each gradient dropped from ``gradients`` once its terms exist. A block's
    gradients come out of its backward together, so they live until the
    block's last layer is scored, and a layer's perturbations are rebuilt from
    its original weight and dropped with their terms; neither is ever held for
    the whole model. The terms and their accumulation are the same operations
    on the same values as when every gradient and perturbation was held, so
    the scores are the same.
    """
    for paths in reversed(by_block):
        for path in paths:
            gradient = gradients.pop(path)
            weight32 = originals[path].astype(mx.float32)
            for width, perturbation in perturbations.items():
                term = mx.sum(mx.abs(gradient * perturbation(path, weight32)))
                accumulated[width][path] = accumulated[width][path] + term
                # One width at a time, so that one rebuilt quantize-dequantize
                # and its temporaries are alive at once rather than all of them.
                mx.eval(accumulated[width][path])


def _scores_from_own_width(
    grad_fn: Any,
    batches: Sequence[mx.array],
    originals: Mapping[str, mx.array],
    *,
    schemes: dict[int, QuantScheme],
    eps: float,
    by_block: Sequence[Sequence[str]],
    report: Callable[[str, int, int], None],
    restore: Callable[[], None],
) -> dict[str, dict[int, float]]:
    """The reference's procedure: each width's backward at its own model.

    One width's quantized copy is resident at a time, the weights the backward
    runs at; its perturbation is that copy less the original, formed per layer
    as its gradient arrives. ``restore`` puts the original weights back in the
    model between widths, since the model's layers hold the last copy they
    were given and the next width's must not be built beside it. See
    MEMORY.md D-036 and M1c.
    """
    paths = [path for block in by_block for path in block]
    scores: dict[str, dict[int, float]] = {path: {} for path in paths}
    passes = len(batches) * len(schemes)
    done = 0
    for width in sorted(schemes):
        anchor = _anchor(originals, schemes[width], eps, _grids(originals, schemes[width]))

        def own(path: str, weight32: mx.array, anchor: dict[str, mx.array] = anchor) -> mx.array:
            return anchor[path] - weight32

        accumulated = {width: {path: mx.zeros((), dtype=mx.float32) for path in paths}}
        for batch in batches:
            _accumulate(grad_fn(anchor, batch), by_block, originals, {width: own}, accumulated)
            done += 1
            report("backward", done, passes)
        for path in paths:
            scores[path][width] = float(accumulated[width][path])
        # The model's layers and the closure hold this width's copy too, and
        # the next width must not find it resident when it builds its own.
        restore()
        del anchor, own
    return scores


def _scores_from_widest(
    grad_fn: Any,
    batches: Sequence[mx.array],
    originals: Mapping[str, mx.array],
    *,
    schemes: dict[int, QuantScheme],
    eps: float,
    by_block: Sequence[Sequence[str]],
    report: Callable[[str, int, int], None],
    restore: Callable[[], None],
) -> dict[str, dict[int, float]]:
    """One backward per batch at the widest candidate, every width scored on it.

    The gradient needs the widest width's quantize-dequantize, which stays
    resident: it is the model the backward runs through. Every width
    contributes its perturbation, and since 2026-10-11 (M1c) none of them is
    held: each width keeps its searched grid, a few numbers per group, and a
    layer's perturbation is rebuilt from its original weight when that layer's
    gradient arrives, then dropped. Before, every width's perturbation was
    resident through the batch loop, one model sized float32 copy per
    candidate, which is what kept a mixed run above 1.5B parameters off a 32
    GB machine. The rebuilt perturbation is the same arithmetic on the same
    values, so the scores are the same; what it costs is the elementwise work
    of one quantize-dequantize per narrower width per batch. See MEMORY.md
    D-036.
    """
    widest = max(schemes)
    grids = {width: _grids(originals, schemes[width]) for width in sorted(schemes)}
    anchor = _anchor(originals, schemes[widest], eps, grids[widest])
    report("perturbations", len(schemes), len(schemes))

    def perturbation_at(width: int) -> Callable[[str, mx.array], mx.array]:
        def perturbation(path: str, weight32: mx.array) -> mx.array:
            if width == widest:
                return anchor[path] - weight32
            return _quantized(weight32, schemes[width], eps, grids[width][path]) - weight32

        return perturbation

    perturbations = {width: perturbation_at(width) for width in sorted(schemes)}
    paths = [path for block in by_block for path in block]
    accumulated = {
        width: {path: mx.zeros((), dtype=mx.float32) for path in paths} for width in schemes
    }
    for index, batch in enumerate(batches, start=1):
        _accumulate(grad_fn(anchor, batch), by_block, originals, perturbations, accumulated)
        report("backward", index, len(batches))
    restore()
    return {
        path: {width: float(accumulated[width][path]) for width in sorted(schemes)}
        for path in paths
    }


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
            at that width. It is this function's default so that a bare call
            measures the control; the entry points in :mod:`mround.api` pass
            ``"widest"``, the measured default (MEMORY.md D-034). ``"widest"``
            takes one backward per batch at the widest candidate and scores
            every width's perturbation against it,
            targeting the mechanism the first measured allocation exposed
            (MEMORY.md D-033): gradients taken through a network the lowest
            width has largely destroyed discriminate poorly and starve the
            late blocks. At the widest width the two modes coincide exactly,
            gradients and perturbations both, which is the invariant the tests
            pin.
        progress: Called with a :class:`ScoringProgress` after the up-front
            perturbations (a stage only ``widest`` has) and after every
            gradient pass. Scoring at the budget
            the reference recommends for 2-bit schemes takes minutes, 323 to
            388 seconds on Qwen2.5-0.5B (rows 39 to 47) and 1512 at the old
            batch of 8 (row 38), and produces no other output, so a caller
            with a console should pass something here.

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
    by_block: list[list[str]] = []
    for block in blocks:
        by_block.append([])
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
            by_block[-1].append(path)
    if not eligible:
        msg = "no layer in any block can be quantized at this group size"
        raise ValueError(msg)

    originals = {path: layer.weight for path, layer in eligible.items()}
    shapes = {
        path: (int(weight.shape[0]), int(weight.shape[1])) for path, weight in originals.items()
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
    # on a 0.5B model. The schedules below hold one quantized copy, the one
    # the backward runs through, rebuild each perturbation when its gradient
    # arrives, and run every block under gradient checkpointing, so that the
    # backward holds one block's activations rather than the model's. See
    # MEMORY.md D-036 and M1c.
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

    def restore() -> None:
        for path, layer in eligible.items():
            layer.weight = originals[path]

    schedule = _scores_from_widest if gradient_source == "widest" else _scores_from_own_width
    try:
        with checkpointed_blocks(model, blocks):
            scores = schedule(
                grad_fn,
                batches,
                originals,
                schemes=schemes,
                eps=eps,
                by_block=by_block,
                report=report,
                restore=restore,
            )
    finally:
        restore()

    return SensitivityScores(
        scores=scores,
        shapes=shapes,
        skipped=tuple(skipped),
        seconds=time.perf_counter() - started,
    )
