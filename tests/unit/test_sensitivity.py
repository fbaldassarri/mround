# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Sensitivity scoring: the score's definition, the cost model, and the handoff.

Three things are pinned here. The DeltaLoss definition itself, against hand
arithmetic, including the property that names it: absolute values are taken
before the sum, so a layer damaged equally in both directions scores its full
damage rather than zero. The storage cost model, against the export path's
accounting (MEMORY.md D-019), because an allocator budgeting with different
arithmetic than the checkpoint reports would produce models that miss their
budget in the direction nobody checks. And the handoff into the allocator,
end to end on an instance small enough to solve by hand.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest

from mround.planner.allocator import allocate_bits
from mround.planner.sensitivity import (
    SENSITIVITY_BATCH_TOKENS,
    SENSITIVITY_SAMPLES,
    SENSITIVITY_SEQ_LEN,
    delta_loss,
    groups_per_row,
    mixed_budget_bits,
    score_layer_options,
    scoring_batch_size,
    storage_bits,
)
from mround.schemes import QuantScheme


class TestDeltaLoss:
    def test_hand_computed(self) -> None:
        weight = [[1.0, -0.5], [0.25, 0.0]]
        quantized = [[0.9, -0.4], [0.35, 0.0]]
        gradient = [[2.0, -3.0], [4.0, 5.0]]
        # |2 * -0.1| + |-3 * 0.1| + |4 * 0.1| + |5 * 0| = 0.2 + 0.3 + 0.4 + 0
        assert delta_loss(weight, quantized, gradient) == pytest.approx(0.9, abs=1e-12)

    def test_absolute_values_come_before_the_sum(self) -> None:
        # The signed first-order term is exactly zero here; the score is the
        # full damage. If an implementation summed first and took the absolute
        # value after, this would come back zero and the property that names
        # the metric would be gone.
        weight = [[0.0, 0.0]]
        quantized = [[1.0, -1.0]]
        gradient = [[1.0, 1.0]]
        assert delta_loss(weight, quantized, gradient) == pytest.approx(2.0, abs=1e-12)

    def test_zero_when_quantization_is_exact(self) -> None:
        weight = np.arange(6.0).reshape(2, 3)
        gradient = np.ones((2, 3)) * 7.0
        assert delta_loss(weight, weight.copy(), gradient) == 0.0

    def test_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="share one shape"):
            delta_loss(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 3)))

    def test_the_scoring_defaults_are_the_measured_ones(self) -> None:
        # 128 sequences of 1024 tokens. These were the reference's own 16x256,
        # adopted for comparability; they are now the budget every measurement
        # in this repository was actually taken at, which is what a default
        # should be. Same model, same seed, same reference checkpoint: 24.2539
        # at the old values against 23.1140 at these, or 4.70 percent, and not
        # one ledger row was produced at the old ones (MEMORY.md D-031).
        # A change here is a deliberate departure, not a tidy-up.
        assert SENSITIVITY_SAMPLES == 128
        assert SENSITIVITY_SEQ_LEN == 1024


class TestScoringBatchSize:
    """The batch size is what keeps a larger scoring budget inside memory."""

    def test_the_default_budget_costs_the_same_activation_peak(self) -> None:
        # The token count per batch is what sets the activation peak, and it is
        # held at 2048 whichever budget runs: the old 16x256 default produced a
        # batch of 8, these produce a batch of 2, and both are 2048 tokens. So
        # moving the default in D-031 bought 4.70 percent of quality without
        # moving the memory the scoring pass needs, which is the whole reason
        # it could move at all.
        assert scoring_batch_size(SENSITIVITY_SAMPLES, SENSITIVITY_SEQ_LEN) == 2
        assert (
            SENSITIVITY_SEQ_LEN * scoring_batch_size(SENSITIVITY_SAMPLES, SENSITIVITY_SEQ_LEN)
            == 256 * 8
        )

    @pytest.mark.parametrize(
        ("seq_len", "expected"),
        [(256, 8), (512, 4), (1024, 2), (2048, 1)],
    )
    def test_tokens_per_batch_stay_constant(self, seq_len: int, expected: int) -> None:
        # The activation peak of a scoring backward follows the token count,
        # not the sequence count, so this is the quantity to hold still. The
        # reference's recommended budget for 2-bit schemes is the 1024 row.
        assert scoring_batch_size(128, seq_len) == expected
        assert expected * seq_len <= SENSITIVITY_BATCH_TOKENS

    def test_a_sequence_longer_than_the_budget_still_runs_one_at_a_time(self) -> None:
        # Refusing here would be wrong: one sequence is the floor of what any
        # backward can do, and the caller asked for a long one deliberately.
        assert scoring_batch_size(8, 4 * SENSITIVITY_BATCH_TOKENS) == 1

    def test_a_draw_smaller_than_the_batch_is_not_padded_up(self) -> None:
        assert scoring_batch_size(3, 256) == 3


class TestStorageBits:
    def test_matches_the_export_accounting(self) -> None:
        # D-019's corrected arithmetic: codes plus one scale and one bias per
        # group at the checkpoint dtype. 4 bits at group size 64 stores 4.50
        # bits per weight, which is the number the export path reports.
        bits = storage_bits((8, 128), 4, 64)
        assert bits == 8 * 128 * 4 + 2 * (8 * 2) * 16
        assert bits / (8 * 128) == pytest.approx(4.5)

    def test_two_bits_at_group_64_stores_two_and_a_half(self) -> None:
        assert storage_bits((4, 64), 2, 64) / (4 * 64) == pytest.approx(2.5)

    def test_metadata_grows_with_smaller_groups(self) -> None:
        coarse = storage_bits((4, 128), 4, 128)
        fine = storage_bits((4, 128), 4, 32)
        assert fine > coarse

    def test_wider_dtype_costs_more(self) -> None:
        assert storage_bits((4, 64), 4, 64, dtype_bits=32) > storage_bits((4, 64), 4, 64)

    def test_indivisible_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="stays dense"):
            storage_bits((4, 100), 4, 64)

    def test_nonpositive_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            storage_bits((0, 64), 4, 64)

    def test_per_channel_costs_one_scale_and_one_bias_per_row(self) -> None:
        # group_size -1 is the scheme's own spelling of per channel scales,
        # and integer arithmetic on -1 used to cost it a negative amount of
        # metadata: in_features % -1 is zero and in_features // -1 is
        # negative. One scale and one bias per output channel is the answer.
        assert storage_bits((4, 64), 4, -1) == 4 * 64 * 4 + 2 * 4 * 16
        assert mixed_budget_bits({"a": (4, 64)}, 3.0, -1) == int(4 * 64 * 3.0) + 2 * 4 * 16
        assert groups_per_row(64, -1) == 1
        assert groups_per_row(64, 32) == 2
        with pytest.raises(ValueError, match="stays dense"):
            groups_per_row(64, 0)


class TestMixedBudget:
    shapes: ClassVar[dict[str, tuple[int, int]]] = {"a": (4, 64), "b": (8, 128)}

    def test_an_integer_average_is_exactly_the_uniform_ceiling(self) -> None:
        # The property that makes "mixed at 3" against "uniform 3" a fair
        # fight: the budget at an integer average equals the exact storage of
        # the uniform allocation at that width, to the bit.
        for width in (2, 3, 4, 8):
            budget = mixed_budget_bits(self.shapes, float(width), 64)
            uniform = sum(storage_bits(shape, width, 64) for shape in self.shapes.values())
            assert budget == uniform

    def test_uniform_at_the_average_is_feasible_under_the_budget(self) -> None:
        # The allocator handed the uniform width as an option must accept it
        # at exactly the budget, never fail by one bit of rounding.
        scheme = QuantScheme(bits=4, group_size=64)
        scores = {name: {2: 1.0, 4: 0.5} for name in self.shapes}
        options = score_layer_options(scores, self.shapes, scheme)
        budget = mixed_budget_bits(self.shapes, 4.0, 64)
        allocation = allocate_bits(options, budget_bits=budget)
        assert allocation.total_cost_bits <= budget
        assert set(allocation.by_layer.values()) <= {2, 4}

    def test_a_fractional_average_sits_between_its_neighbours(self) -> None:
        low = mixed_budget_bits(self.shapes, 2.0, 64)
        mid = mixed_budget_bits(self.shapes, 2.5, 64)
        high = mixed_budget_bits(self.shapes, 3.0, 64)
        assert low < mid < high

    def test_empty_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            mixed_budget_bits({}, 3.0, 64)

    def test_a_nonpositive_average_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            mixed_budget_bits(self.shapes, 0.0, 64)

    def test_an_ungroupable_shape_is_refused(self) -> None:
        with pytest.raises(ValueError, match="grouped"):
            mixed_budget_bits({"a": (4, 100)}, 3.0, 64)


class TestScoreLayerOptions:
    scheme = QuantScheme(bits=4, group_size=64)

    def test_end_to_end_with_the_allocator(self) -> None:
        # Solvable by hand. Costs per layer: 2 bits is 4*64*2 + 2*4*16 = 640,
        # 4 bits is 1024 + 128 = 1152. A budget of exactly 640 + 1152 = 1792
        # affords one layer at each width, and the sensitive layer must be the
        # one that gets 4 bits.
        shapes = {"a": (4, 64), "b": (4, 64)}
        scores = {"a": {2: 100.0, 4: 1.0}, "b": {2: 5.0, 4: 0.5}}
        options = score_layer_options(scores, shapes, self.scheme)
        allocation = allocate_bits(options, budget_bits=1792)
        assert allocation.by_layer == {"a": 4, "b": 2}
        assert allocation.total_cost_bits == 1792

    def test_options_carry_the_cost_model(self) -> None:
        options = score_layer_options({"a": {2: 1.0, 4: 0.5}}, {"a": (4, 64)}, self.scheme)
        by_bits = {option.bits: option.cost_bits for option in options["a"]}
        assert by_bits == {2: 640, 4: 1152}

    def test_widths_come_out_ascending(self) -> None:
        options = score_layer_options({"a": {8: 0.1, 2: 3.0, 4: 1.0}}, {"a": (4, 64)}, self.scheme)
        assert [option.bits for option in options["a"]] == [2, 4, 8]

    def test_a_single_width_pins_the_layer(self) -> None:
        # A layer scored at one width has no choice, which is how a pinned
        # layer is expressed to the allocator.
        options = score_layer_options({"a": {4: 1.0}}, {"a": (4, 64)}, self.scheme)
        allocation = allocate_bits(options, budget_bits=10_000)
        assert allocation.by_layer == {"a": 4}

    def test_layers_must_match_between_scores_and_shapes(self) -> None:
        with pytest.raises(ValueError, match="same layers"):
            score_layer_options({"a": {4: 1.0}}, {"b": (4, 64)}, self.scheme)

    def test_every_layer_must_share_one_width_set(self) -> None:
        with pytest.raises(ValueError, match="same scoring passes"):
            score_layer_options(
                {"a": {2: 1.0, 4: 1.0}, "b": {4: 1.0}},
                {"a": (4, 64), "b": (4, 64)},
                self.scheme,
            )

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0])
    def test_invalid_scores_are_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="invalid score"):
            score_layer_options({"a": {4: bad}}, {"a": (4, 64)}, self.scheme)

    def test_empty_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no layers"):
            score_layer_options({}, {}, self.scheme)

    def test_a_layer_with_no_widths_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no scored widths"):
            score_layer_options({"a": {}}, {"a": (4, 64)}, self.scheme)
