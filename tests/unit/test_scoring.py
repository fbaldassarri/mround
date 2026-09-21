# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The MLX sensitivity scorer, against finite differences and the NumPy definition.

The decisive test here is the first one, and it makes two separable claims.
That the gradient of the calibration loss at the quantized weights is what
``mx.grad`` says it is, checked by directional finite differences, arithmetic
that shares nothing with autodiff. And that the scorer's number is exactly the
NumPy DeltaLoss definition applied to that gradient, which pins the gradient
node, the layer mapping, and the reduction. The first version of this oracle
differenced the loss entrywise and pushed the noisy result through the
absolute-value reduction, which turns zero-mean noise into a one-sided bias;
log0031 recorded it failing high by seven percent. The second version fixed
that and still failed on Metal (log0033), because the GPU matmul kernel
truncates its inputs to roughly float16 precision (MEMORY.md D-015) and
finite differences through it are granularity-limited at any useful step; the
oracle now runs on the CPU stream, whose matmul is honest float32. The
comments inside the test carry both mechanisms, because they are the kind of
mistake that gets remade.

A toy model stands in for a real one: an embedding, a discoverable block
stack, and a head outside the blocks. That shape is what lets the same
discovery path the block loop uses find the stack, and what pins the scoring
boundary: the head and the embedding stay full precision and unscored.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="MLX is not installed")

import mlx.core as mx
from mlx import nn

from mround.core.quantizer import fake_quantize
from mround.core.scale_search import search_scales
from mround.pipeline.scoring import ScoringProgress, score_widths
from mround.planner.sensitivity import delta_loss
from mround.schemes import QuantScheme, ScaleInit

VOCAB = 11
DIM = 8
GROUP = 4


class Block(nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(DIM, DIM, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        return h + self.up(h)


class Toy(nn.Module):  # type: ignore[misc]
    """Embedding, two discoverable blocks, and a head outside the stack."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, DIM)
        self.layers = [Block(), Block()]
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class MixedBlock(nn.Module):  # type: ignore[misc]
    """A block holding one groupable linear and one that cannot group."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(DIM, DIM, bias=False)
        self.odd = nn.Linear(6, DIM, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        return self.up(h) + self.odd(h[..., :6])


class MixedToy(Toy):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [MixedBlock(), MixedBlock()]


def batch_of_tokens(seed: int = 0, rows: int = 2, tokens: int = 6) -> mx.array:
    rng = np.random.default_rng(seed)
    return mx.array(rng.integers(0, VOCAB, size=(rows, tokens)))


def causal_loss(model: nn.Module, batch: mx.array) -> float:
    """The scorer's loss convention, recomputed independently for the oracle."""
    logits = model(batch[:, :-1]).astype(mx.float32)
    losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
    return float(mx.mean(losses))


class TestScoreWidths:
    def test_matches_finite_differences_and_the_numpy_definition(self) -> None:
        # Two steps, because the first version of this oracle was biased and
        # blamed the scorer (log0031). It differenced the float32 loss one
        # entry at a time, which puts noise of order 1e-4 on every gradient
        # entry, then pushed that gradient through the absolute-value
        # reduction, where |g + noise| inflates every entry whose true |g|
        # sits below the noise. Zero-mean noise becomes a one-sided bias, and
        # the oracle reads several percent high. An unbiased estimator fed
        # through an absolute value is not unbiased any more.
        #
        # Step one validates the gradient itself with directional finite
        # differences: one scalar difference per direction, so the noise is
        # divided by the step once rather than once per entry, and the
        # gradient's own direction carries maximal signal. Step two checks the
        # scorer's number against the NumPy DeltaLoss of an independently
        # computed gradient, to one part in ten thousand, with no finite
        # differences in the path at all. Together they pin the gradient node,
        # the layer mapping, and the reduction; separately, a failure names
        # which half is wrong.
        #
        # The whole test is pinned to the CPU stream, deliberately, and this
        # is the second lesson this oracle taught (log0033). On this
        # project's GPU the matmul kernel truncates its inputs to roughly
        # float16 precision (MEMORY.md D-015), so a perturbation of order
        # 1e-3 per entry spans only a few truncation granules and the
        # difference quotient inherits errors of several percent at every
        # step size in the useful range; the same test passes on the CPU
        # backend and failed by eight percent on Metal. Finite differences
        # validate gradient semantics, and semantics are stream-independent;
        # the GPU kernel's precision is a separate, measured property with
        # its own probes.
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        layers: dict[str, Any] = {}
        originals: dict[str, Any] = {}
        try:
            model = Toy()
            batch = batch_of_tokens()
            base = QuantScheme(bits=4, group_size=GROUP)
            result = score_widths(model, [batch], [2, 4], base)

            layers = {"layers.0.up": model.layers[0].up, "layers.1.up": model.layers[1].up}
            assert set(result.scores) == set(layers)
            assert result.shapes == dict.fromkeys(layers, (DIM, DIM))

            originals = {path: layer.weight for path, layer in layers.items()}

            def swap_and_loss(assignments: dict[str, mx.array]) -> mx.array:
                for path, layer in layers.items():
                    layer.weight = assignments[path]
                logits = model(batch[:, :-1]).astype(mx.float32)
                return mx.mean(nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none"))

            for width in (2, 4):
                scheme = dataclasses.replace(base, bits=width)
                quantized = {
                    path: fake_quantize(weight.astype(mx.float32), None, scheme)
                    for path, weight in originals.items()
                }
                mx.eval(list(quantized.values()))

                gradients = mx.grad(swap_and_loss)(quantized)
                mx.eval(list(gradients.values()))

                # Step one: the gradient is real. The first direction is the
                # gradient's own, where the projection equals the gradient
                # norm and dwarfs the differencing noise; the second is
                # random, constraining orientation. The absolute floor covers
                # a random projection that happens to land near zero.
                step = 1e-2
                rng = np.random.default_rng(width)
                for path in layers:
                    gradient = np.array(gradients[path], dtype=np.float64)
                    aligned = gradient / np.linalg.norm(gradient)
                    random = rng.normal(size=(DIM, DIM))
                    random /= np.linalg.norm(random)
                    for direction in (aligned, random):
                        offset = mx.array((direction * step).astype(np.float32))
                        nudged_up = dict(quantized)
                        nudged_up[path] = quantized[path] + offset
                        nudged_down = dict(quantized)
                        nudged_down[path] = quantized[path] - offset
                        fd = (
                            float(swap_and_loss(nudged_up)) - float(swap_and_loss(nudged_down))
                        ) / (2 * step)
                        analytic = float(np.sum(gradient * direction))
                        assert analytic == pytest.approx(fd, rel=3e-2, abs=1e-4)

                # Step two: the scorer's reduction is the definition.
                for path in layers:
                    expected = delta_loss(
                        np.array(originals[path].astype(mx.float32)),
                        np.array(quantized[path]),
                        np.array(gradients[path]),
                    )
                    got = result.scores[path][width]
                    assert got == pytest.approx(expected, rel=1e-4, abs=1e-8)
        finally:
            mx.set_default_device(previous_device)
            for path, layer in layers.items():
                layer.weight = originals[path]

    def test_the_model_comes_back_exactly_as_it_went_in(self) -> None:
        model = Toy()
        batch = batch_of_tokens(seed=3)
        before = {"0": model.layers[0].up.weight, "1": model.layers[1].up.weight}
        loss_before = causal_loss(model, batch)
        score_widths(model, [batch], [2], QuantScheme(bits=2, group_size=GROUP))
        assert model.layers[0].up.weight is before["0"]
        assert model.layers[1].up.weight is before["1"]
        assert causal_loss(model, batch) == pytest.approx(loss_before, rel=0, abs=0)

    def test_scores_accumulate_over_batches(self) -> None:
        # Two batches must score the sum of each batch alone: the reference
        # accumulates per-batch sums, and a mean would silently change the
        # metric's scale with the calibration size.
        model = Toy()
        first, second = batch_of_tokens(seed=1), batch_of_tokens(seed=2)
        base = QuantScheme(bits=3, group_size=GROUP)
        one = score_widths(model, [first], [3], base).scores
        two = score_widths(model, [second], [3], base).scores
        both = score_widths(model, [first, second], [3], base).scores
        for path in both:
            assert both[path][3] == pytest.approx(one[path][3] + two[path][3], rel=1e-5)

    def test_ungroupable_layers_are_reported_not_scored(self) -> None:
        model = MixedToy()
        result = score_widths(
            model, [batch_of_tokens()], [4], QuantScheme(bits=4, group_size=GROUP)
        )
        assert set(result.scores) == {"layers.0.up", "layers.1.up"}
        skipped = dict(result.skipped)
        assert set(skipped) == {"layers.0.odd", "layers.1.odd"}
        assert "not a multiple" in skipped["layers.0.odd"]

    def test_a_searched_scheme_is_scored_on_the_searched_grid(self) -> None:
        # The score exists to predict the damage the actual quantizer will do,
        # and at 2 bits MRound's actual quantizer is the searched one. Scoring
        # the searched scheme must therefore produce different numbers than the
        # observed grid does, and the difference must come from the same search
        # the quantizer runs (D-031).
        model = Toy()
        batch = batch_of_tokens(seed=5)
        observed = score_widths(model, [batch], [2], QuantScheme(bits=2, group_size=GROUP))
        searched = score_widths(
            model,
            [batch],
            [2],
            QuantScheme(bits=2, group_size=GROUP, scale_init=ScaleInit.SEARCHED),
        )
        weight = model.layers[0].up.weight.astype(mx.float32)
        scheme = QuantScheme(bits=2, group_size=GROUP, scale_init=ScaleInit.SEARCHED)
        grid_moved = bool(
            mx.any(
                fake_quantize(weight, None, scheme, init_scale=search_scales(weight, scheme))
                != fake_quantize(weight, None, QuantScheme(bits=2, group_size=GROUP))
            )
        )
        assert grid_moved
        differs = any(
            searched.scores[path][2] != pytest.approx(observed.scores[path][2], rel=1e-9)
            for path in searched.scores
        )
        assert differs

    def test_the_scores_drive_a_feasible_allocation(self) -> None:
        # The whole chain below the API: score, assemble options, budget at a
        # fractional average, allocate. What must hold on any model: every
        # chosen width is a candidate, the layer set is preserved, the cost
        # respects the budget, and the achieved code average does not exceed
        # the target.
        from mround.planner.allocator import allocate_bits  # noqa: PLC0415
        from mround.planner.sensitivity import (  # noqa: PLC0415
            mixed_budget_bits,
            score_layer_options,
        )

        model = Toy()
        base = QuantScheme(bits=4, group_size=GROUP)
        scored = score_widths(model, [batch_of_tokens(seed=9)], [2, 4], base)
        options = score_layer_options(scored.scores, scored.shapes, base)
        budget = mixed_budget_bits(scored.shapes, 3.0, GROUP)
        allocation = allocate_bits(options, budget_bits=budget)

        assert set(allocation.by_layer) == set(scored.scores)
        assert set(allocation.by_layer.values()) <= {2, 4}
        assert allocation.total_cost_bits <= budget
        elements = sum(out * inner for out, inner in scored.shapes.values())
        metadata = sum(2 * out * (inner // GROUP) * 16 for out, inner in scored.shapes.values())
        achieved = (allocation.total_cost_bits - metadata) / elements
        assert achieved <= 3.0

    def test_widest_gradients_coincide_with_own_width_at_the_widest(self) -> None:
        # The invariant that makes the two modes one family: at the widest
        # candidate the gradient is taken at the same quantized model and the
        # perturbation is the same tensor, so the scores must agree. A
        # disagreement here means the modes differ by more than their stated
        # difference.
        model = Toy()
        batch = batch_of_tokens(seed=11)
        base = QuantScheme(bits=4, group_size=GROUP)
        own = score_widths(model, [batch], [2, 4], base, gradient_source="own_width")
        widest = score_widths(model, [batch], [2, 4], base, gradient_source="widest")
        for path in own.scores:
            assert widest.scores[path][4] == pytest.approx(own.scores[path][4], rel=1e-6)

    def test_widest_gradients_differ_at_the_narrow_width(self) -> None:
        # And at the narrow width they must differ, because that is the entire
        # point: the gradients come from an intact model instead of one the
        # narrow grid damaged. Identical scores would mean the mode is wired
        # to nothing.
        model = Toy()
        batch = batch_of_tokens(seed=12)
        base = QuantScheme(bits=4, group_size=GROUP)
        own = score_widths(model, [batch], [2, 4], base, gradient_source="own_width")
        widest = score_widths(model, [batch], [2, 4], base, gradient_source="widest")
        assert any(
            widest.scores[path][2] != pytest.approx(own.scores[path][2], rel=1e-9)
            for path in own.scores
        )

    def test_progress_reports_every_backward_under_widest_gradients(self) -> None:
        # A scoring pass at the budget the reference recommends for 2-bit
        # schemes runs for over an hour and used to print nothing at all, which
        # is how log0059 came to be reported as a hang. The contract is one
        # report per gradient pass plus one when the perturbations are built.
        seen: list[Any] = []
        score_widths(
            Toy(),
            [batch_of_tokens(seed=1), batch_of_tokens(seed=2), batch_of_tokens(seed=3)],
            [2, 4],
            QuantScheme(bits=4, group_size=GROUP),
            gradient_source="widest",
            progress=seen.append,
        )
        assert [(step.stage, step.index, step.total) for step in seen] == [
            ("perturbations", 2, 2),
            ("backward", 1, 3),
            ("backward", 2, 3),
            ("backward", 3, 3),
        ]
        # Monotone in time, and the projection is the linear extrapolation a
        # reader is meant to plan around: it extrapolates the backward loop
        # from the loop's own clock, not from the start of scoring, because
        # under widest the perturbation stage runs first and would otherwise
        # inflate every projection until the last batch.
        assert seen[-1].seconds >= seen[0].seconds
        assert seen[1].stage_seconds <= seen[1].seconds
        assert seen[1].projected_seconds == pytest.approx(seen[1].stage_seconds * 3)
        assert seen[-1].remaining_seconds == pytest.approx(0.0)

    def test_the_projection_starts_its_clock_where_the_stage_starts(self) -> None:
        # Sixty seconds of perturbations followed by ten second batches: the
        # old projection said 4480s at the first of 64 batches against a true
        # 640s for the loop. The stage clock says 640 and 630 remaining.
        step = ScoringProgress(
            stage="backward",
            index=1,
            total=64,
            seconds=70.0,
            peak_memory_bytes=0,
            stage_seconds=10.0,
        )
        assert step.projected_seconds == pytest.approx(640.0)
        assert step.remaining_seconds == pytest.approx(630.0)
        assert "about 630s remaining" in step.describe()
        assert "peak" not in step.describe()

    def test_progress_counts_every_width_under_own_width_gradients(self) -> None:
        # Under the reference's own schedule the work is the batch count times
        # the candidate count, and the total has to say so or the projection
        # printed after the first width would be three times optimistic.
        seen: list[Any] = []
        score_widths(
            Toy(),
            [batch_of_tokens(seed=4), batch_of_tokens(seed=5)],
            [2, 3, 4],
            QuantScheme(bits=4, group_size=GROUP),
            gradient_source="own_width",
            progress=seen.append,
        )
        assert [(step.stage, step.index, step.total) for step in seen] == [
            ("backward", index, 6) for index in range(1, 7)
        ]

    def test_a_progress_line_names_the_stage_and_the_peak(self) -> None:
        seen: list[Any] = []
        score_widths(
            Toy(),
            [batch_of_tokens(seed=6)],
            [4],
            QuantScheme(bits=4, group_size=GROUP),
            gradient_source="widest",
            progress=seen.append,
        )
        assert "perturbations for 1 widths" in seen[0].describe()
        assert "backward 1/1" in seen[-1].describe()
        # The peak is printed only when the build reports one, and then in
        # decimal gigabytes like every other memory figure in the project.
        with_peak = ScoringProgress(
            stage="backward", index=1, total=1, seconds=2.0, peak_memory_bytes=2_500_000_000
        )
        assert "peak memory 2.50 GB" in with_peak.describe()

    def test_scoring_without_a_progress_callback_still_runs(self) -> None:
        # The callback is optional, and the oracle tests above rely on it
        # staying that way.
        result = score_widths(
            Toy(), [batch_of_tokens(seed=7)], [4], QuantScheme(bits=4, group_size=GROUP)
        )
        assert result.scores

    def test_an_unknown_gradient_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="gradient_source"):
            score_widths(
                Toy(),
                [batch_of_tokens()],
                [4],
                QuantScheme(bits=4, group_size=GROUP),
                gradient_source="fp",
            )

    def test_empty_widths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="no candidate widths"):
            score_widths(Toy(), [batch_of_tokens()], [], QuantScheme(bits=4, group_size=GROUP))

    def test_a_group_size_nothing_fits_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no layer in any block"):
            score_widths(Toy(), [batch_of_tokens()], [4], QuantScheme(bits=4, group_size=5))
