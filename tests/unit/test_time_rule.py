# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The rule for time, held to every run it was fitted on and to what it promises.

``mround/planner/timing.py`` predicts the seconds a quantization run records
(ROAD-TO-1.0.md, M2), on the one machine the ledger's runs were made on. What
it promises is the runs it reproduces, so those are written out here as that
Mac recorded them: fifty five runs at the standard recipe on the trees since
2026-10-12, each with its log, its configuration, its recorded seconds
(``result.total_seconds`` in ``mround_run.json``), a mixed run's scoring pass
(``mixed.scoring_seconds``) and the 25th percentile of its block lines, the
pace of the machine undisturbed. The rule is held to each: every run between
:data:`~mround.planner.timing.SPREAD`'s two ends of its prediction, the median
within two percent, 49 of the 55 within fifteen and the other six over it, and
every run's blocks within fifteen percent. A refit that loses one fails here.

The rest is what the rule has to do whatever its constants: grow with the
steps and the pieces, drop the search when there is none, add a scoring pass
only where one runs, and resolve the outlier rule and the scoring gradients
as the run does.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Any

import pytest

from mround.exceptions import SchemeError
from mround.planner.memory import ModelShape, RunShape
from mround.planner.timing import (
    MACHINE,
    RULE_VERSION,
    SPREAD,
    TimePrediction,
    predict_run_time,
    predict_time,
)
from mround.planner.timing import _duration as duration
from mround.schemes import TuningConfig

QWEN: dict[str, Any] = {"model_type": "qwen2", "torch_dtype": "bfloat16", "vocab_size": 151936}
CONFIGS: dict[str, dict[str, Any]] = {
    "smollm2": {
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "hidden_size": 576,
        "intermediate_size": 1536,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "vocab_size": 49152,
        "tie_word_embeddings": True,
    },
    "qwen0.5b": {
        **QWEN,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
    },
    "qwen1.5b": {
        **QWEN,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
    },
    "qwen3b": {
        **QWEN,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
    },
    "qwen7b": {
        **QWEN,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
    },
}


def _shape(model: str) -> ModelShape:
    shape = ModelShape.from_config(CONFIGS[model])
    assert shape is not None
    return shape


@dataclasses.dataclass(frozen=True)
class Run:
    """One run as the Mac recorded it, at the standard recipe and group size 64.

    Attributes:
        log: The log its block lines are in.
        model: A key of :data:`CONFIGS`.
        kind: ``"2 bits rule"`` at the default, where the outlier rule is
            on; ``"2 bits"`` with it turned off; ``"4 bits"``; ``"mixed
            widest"`` and ``"mixed own_width"`` at mixed 2.5 with their
            scoring pass; ``"mixed, plan given"`` without one.
        pieces: Pieces each step ran in.
        seconds: ``result.total_seconds``, the block loop with its entrance.
        scoring: ``mixed.scoring_seconds``, zero without a scoring pass.
        block: The 25th percentile of the run's block lines, in seconds.
    """

    log: int
    model: str
    kind: str
    pieces: int
    seconds: float
    scoring: float
    block: float

    def __str__(self) -> str:
        """The test id."""
        return f"log{self.log:04d}"

    def predicted(self) -> TimePrediction:
        """What the rule says this run takes."""
        mixed = self.kind.startswith("mixed") and self.scoring > 0
        run = RunShape(micro_batches=self.pieces, mixed=mixed)
        return predict_time(
            _shape(self.model),
            run,
            iters=200,
            outlier_rule=self.kind == "2 bits rule",
            scoring_source="own_width" if self.kind == "mixed own_width" else "widest",
        )

    @property
    def recorded(self) -> float:
        """Every second the run recorded, its scoring pass included."""
        return self.seconds + self.scoring


# Read from each run's mround_run.json and its log's block lines on the Mac,
# 2026-11-13. Logs 0157 to 0159 are Q1's 3B arms; 0170 to 0242 the 7B in four
# pieces at seeds 42 to 44 and the learning rate's A/B; 0187 to 0202 the
# plain loss; 0251 to 0335 the mixed tables; 0341 to 0349 M2's 4 bit runs.
RUNS = [
    Run(157, "qwen3b", "2 bits rule", 1, 16561.5, 0.0, 442.9),
    Run(158, "qwen3b", "2 bits rule", 1, 16497.2, 0.0, 441.8),
    Run(159, "qwen3b", "2 bits rule", 1, 16546.4, 0.0, 445.7),
    Run(170, "qwen7b", "2 bits rule", 4, 59345.5, 0.0, 2062.0),
    Run(177, "qwen7b", "2 bits rule", 4, 59702.1, 0.0, 2078.2),
    Run(180, "qwen7b", "2 bits rule", 4, 59826.3, 0.0, 2079.3),
    Run(187, "qwen3b", "2 bits", 1, 17325.6, 0.0, 425.1),
    Run(191, "qwen7b", "2 bits", 4, 50516.7, 0.0, 1706.8),
    Run(195, "qwen7b", "2 bits", 4, 42638.5, 0.0, 1462.8),
    Run(199, "qwen3b", "2 bits", 1, 14857.4, 0.0, 387.1),
    Run(202, "qwen3b", "2 bits", 1, 14695.7, 0.0, 387.4),
    Run(205, "qwen3b", "2 bits rule", 1, 17046.3, 0.0, 444.7),
    Run(208, "qwen7b", "2 bits rule", 4, 62731.5, 0.0, 2181.4),
    Run(212, "qwen3b", "2 bits rule", 1, 18176.8, 0.0, 487.6),
    Run(215, "qwen3b", "2 bits rule", 1, 18381.7, 0.0, 492.8),
    Run(218, "qwen0.5b", "2 bits rule", 1, 4066.4, 0.0, 164.0),
    Run(219, "qwen0.5b", "2 bits rule", 1, 3928.6, 0.0, 160.4),
    Run(223, "qwen0.5b", "2 bits rule", 1, 3880.6, 0.0, 159.1),
    Run(226, "qwen0.5b", "2 bits rule", 1, 3912.6, 0.0, 158.3),
    Run(229, "qwen1.5b", "2 bits rule", 1, 9432.7, 0.0, 326.9),
    Run(232, "qwen1.5b", "2 bits rule", 1, 9410.6, 0.0, 327.2),
    Run(235, "qwen1.5b", "2 bits rule", 1, 9461.1, 0.0, 327.2),
    Run(238, "qwen7b", "2 bits rule", 4, 63421.4, 0.0, 2211.7),
    Run(242, "qwen7b", "2 bits rule", 4, 63539.4, 0.0, 2195.1),
    Run(251, "qwen3b", "mixed widest", 1, 16758.3, 2961.2, 410.8),
    Run(254, "qwen3b", "mixed widest", 1, 15408.6, 3313.3, 411.0),
    Run(257, "qwen3b", "mixed widest", 1, 17114.4, 3176.8, 436.0),
    Run(260, "qwen3b", "mixed widest", 1, 19517.3, 4814.5, 494.1),
    Run(264, "qwen3b", "mixed widest", 1, 18374.0, 3052.6, 495.0),
    Run(267, "qwen3b", "mixed widest", 1, 18345.5, 3051.9, 493.1),
    Run(274, "qwen3b", "mixed, plan given", 1, 16857.3, 0.0, 440.8),
    Run(278, "qwen3b", "mixed, plan given", 1, 16245.0, 0.0, 436.5),
    Run(281, "qwen3b", "mixed, plan given", 1, 16123.3, 0.0, 434.4),
    Run(288, "qwen3b", "mixed own_width", 1, 15571.7, 4009.1, 416.9),
    Run(292, "qwen3b", "mixed own_width", 1, 16086.0, 4409.4, 430.6),
    Run(295, "qwen3b", "mixed own_width", 1, 19832.4, 4741.8, 441.8),
    Run(298, "smollm2", "mixed own_width", 1, 1872.2, 268.2, 61.6),
    Run(300, "smollm2", "mixed widest", 1, 1851.4, 149.2, 60.7),
    Run(302, "smollm2", "mixed own_width", 1, 1873.1, 265.7, 61.6),
    Run(304, "smollm2", "mixed widest", 1, 1862.3, 152.5, 60.9),
    Run(306, "smollm2", "mixed own_width", 1, 1855.8, 262.7, 61.1),
    Run(308, "smollm2", "mixed widest", 1, 1879.2, 149.8, 61.8),
    Run(310, "qwen0.5b", "mixed own_width", 1, 3582.0, 612.1, 142.8),
    Run(313, "qwen0.5b", "mixed widest", 1, 3677.7, 472.4, 144.2),
    Run(316, "qwen0.5b", "mixed own_width", 1, 3506.2, 602.5, 142.1),
    Run(319, "qwen0.5b", "mixed widest", 1, 3514.3, 415.7, 142.6),
    Run(322, "qwen0.5b", "mixed own_width", 1, 3906.7, 693.6, 147.2),
    Run(325, "qwen0.5b", "mixed widest", 1, 3596.6, 420.2, 144.9),
    Run(328, "smollm2", "mixed widest", 1, 1917.9, 151.5, 62.5),
    Run(331, "qwen0.5b", "mixed widest", 1, 3549.7, 433.0, 143.7),
    Run(335, "qwen3b", "mixed own_width", 1, 17671.5, 4089.4, 438.5),
    Run(341, "qwen7b", "4 bits", 4, 57008.5, 0.0, 1717.5),
    Run(343, "qwen0.5b", "4 bits", 1, 4464.5, 0.0, 163.9),
    Run(347, "qwen1.5b", "4 bits", 1, 10409.8, 0.0, 349.2),
    Run(349, "qwen3b", "4 bits", 1, 21643.1, 0.0, 423.5),
]


def _ratio(run: Run) -> float:
    return run.recorded / run.predicted().total


class TestEveryRun:
    """The fifty five runs the rule was fitted on, each against it."""

    def test_there_are_fifty_five(self) -> None:
        assert len(RUNS) == 55
        assert len({run.log for run in RUNS}) == 55

    @pytest.mark.parametrize("run", RUNS, ids=str)
    def test_every_run_falls_within_the_spread(self, run: Run) -> None:
        assert SPREAD[0] - 0.001 <= _ratio(run) <= SPREAD[1] + 0.001

    def test_the_spread_is_the_runs_own(self) -> None:
        ratios = [_ratio(run) for run in RUNS]
        assert min(ratios) == pytest.approx(SPREAD[0], abs=0.001)
        assert max(ratios) == pytest.approx(SPREAD[1], abs=0.001)

    def test_the_median_run_is_within_two_percent(self) -> None:
        assert abs(statistics.median(_ratio(run) for run in RUNS) - 1) < 0.02

    def test_forty_nine_of_them_are_within_fifteen_percent(self) -> None:
        # The six outside are slow stretches of the machine, all over the rule.
        within = [run for run in RUNS if 0.85 <= _ratio(run) <= 1.15]
        assert len(within) >= 49
        assert all(_ratio(run) > 1.15 for run in RUNS if run not in within)

    @pytest.mark.parametrize("run", RUNS, ids=str)
    def test_every_run_s_blocks_are_within_fifteen_percent(self, run: Run) -> None:
        # The 25th percentile of a run's block lines, against the line the
        # rule predicts: the machine's pace, which the slow stretches above it
        # do not move.
        assert 0.85 <= run.block / run.predicted().block <= 1.15

    def test_the_same_work_on_two_days_is_the_machine(self) -> None:
        # Qwen2.5-0.5B-Instruct at 4 bits wrote the same checkpoint to the
        # byte in September (log0109, 3,606.0 s) and on 2026-11-12 (log0343):
        # one inside a few percent of the rule, the other a quarter over it.
        september = dataclasses.replace(RUNS[-3], log=109, seconds=3606.0, block=144.2)
        assert RUNS[-3].log == 343
        assert _ratio(september) == pytest.approx(0.98, abs=0.01)
        assert _ratio(RUNS[-3]) == pytest.approx(1.21, abs=0.01)


class TestWhatTheRuleSays:
    """Its shape, whatever its constants."""

    def test_the_steps_scale_the_blocks_and_nothing_else(self) -> None:
        model, run = _shape("qwen3b"), RunShape()
        standard = predict_time(model, run, iters=200, outlier_rule=False)
        longer = predict_time(model, run, iters=1000, outlier_rule=False)
        searched = 0.307e-6 * model.block_weights
        assert (longer.block - searched) / (standard.block - searched) == pytest.approx(1010 / 210)
        assert longer.stages["between_blocks"] == standard.stages["between_blocks"]

    def test_pieces_cost_time_and_the_outlier_rule_more_in_pieces(self) -> None:
        model = _shape("qwen7b")
        costs = {}
        for pieces in (1, 4):
            run = RunShape(micro_batches=pieces)
            plain = predict_time(model, run, iters=200, outlier_rule=False)
            ruled = predict_time(model, run, iters=200, outlier_rule=True)
            costs[pieces] = (plain.block, ruled.block / plain.block)
        assert costs[4][0] > costs[1][0]
        assert costs[4][1] > costs[1][1] > 1

    def test_without_the_search_a_block_loses_exactly_the_search(self) -> None:
        model = _shape("qwen1.5b")
        searched = predict_time(model, RunShape(searched=True), iters=200, outlier_rule=False)
        bare = predict_time(model, RunShape(searched=False), iters=200, outlier_rule=False)
        assert searched.block - bare.block == pytest.approx(0.307e-6 * model.block_weights)

    def test_only_a_mixed_run_scores(self) -> None:
        model = _shape("qwen0.5b")
        uniform = predict_time(model, RunShape(mixed=False), iters=200, outlier_rule=False)
        mixed = predict_time(model, RunShape(mixed=True), iters=200, outlier_rule=False)
        assert "scoring" not in uniform.stages
        assert mixed.stages["scoring"] > 0
        assert mixed.total - uniform.total == pytest.approx(mixed.stages["scoring"])

    def test_a_backward_at_each_width_costs_more(self) -> None:
        model, run = _shape("qwen3b"), RunShape(mixed=True)
        widest = predict_time(model, run, iters=200, outlier_rule=False, scoring_source="widest")
        own = predict_time(model, run, iters=200, outlier_rule=False, scoring_source="own_width")
        assert own.stages["scoring"] / widest.stages["scoring"] == pytest.approx(1.42)

    def test_the_scoring_pass_scales_with_its_tokens(self) -> None:
        model, run = _shape("qwen3b"), RunShape(mixed=True)
        default = predict_time(model, run, iters=200, outlier_rule=False)
        double = predict_time(model, run, iters=200, outlier_rule=False, scoring_tokens=262144)
        assert double.stages["scoring"] == pytest.approx(2 * default.stages["scoring"])
        assert default.scoring_tokens == 128 * 1024

    def test_a_larger_model_takes_longer(self) -> None:
        totals = [
            predict_time(_shape(name), RunShape(), iters=200, outlier_rule=False).total
            for name in ("smollm2", "qwen0.5b", "qwen1.5b", "qwen3b", "qwen7b")
        ]
        assert totals == sorted(totals)

    def test_a_step_count_that_is_not_positive_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive number of steps"):
            predict_time(_shape("qwen0.5b"), RunShape(), iters=0, outlier_rule=False)


class TestResolution:
    """The settings as the run resolves them (``predict_run_time``)."""

    @pytest.mark.parametrize(("bits", "expected"), [(2, True), (3, True), (4, False), (8, False)])
    def test_the_outlier_rule_follows_the_base_width(self, bits: int, expected: bool) -> None:
        prediction = predict_run_time(_shape("qwen0.5b"), RunShape(), TuningConfig(), bits)
        assert prediction.outlier_rule is expected

    def test_a_rule_turned_off_is_off(self) -> None:
        tuning = TuningConfig(suppress_outliers=False)
        assert not predict_run_time(_shape("qwen0.5b"), RunShape(), tuning, 2).outlier_rule

    def test_a_mixed_run_tunes_at_its_base_width(self) -> None:
        # Every published mixed run: base 4, so the plain loss in every block.
        run = RunShape(mixed=True)
        prediction = predict_run_time(
            _shape("qwen3b"), run, TuningConfig(), 4, candidate_bits=(2, 3, 4)
        )
        assert prediction.outlier_rule is False

    def test_a_per_layer_recipe_takes_the_narrowest_width(self) -> None:
        run = RunShape(mixed=True)
        prediction = predict_run_time(
            _shape("qwen3b"),
            run,
            TuningConfig(),
            4,
            candidate_bits=(2, 3, 4),
            per_layer_recipe=True,
        )
        assert prediction.outlier_rule is True

    @pytest.mark.parametrize(
        ("model", "expected"), [("qwen0.5b", "widest"), ("qwen3b", "own_width")]
    )
    def test_the_scoring_gradients_follow_the_size_rule(self, model: str, expected: str) -> None:
        run = RunShape(mixed=True)
        prediction = predict_run_time(_shape(model), run, TuningConfig(), 4, candidate_bits=(2, 4))
        assert prediction.scoring_source == expected

    def test_a_named_source_is_taken(self) -> None:
        run = RunShape(mixed=True)
        prediction = predict_run_time(
            _shape("qwen3b"), run, TuningConfig(), 4, gradient_source="widest"
        )
        assert prediction.scoring_source == "widest"

    def test_an_unknown_source_is_refused_for_a_mixed_run_alone(self) -> None:
        with pytest.raises(SchemeError, match="sensitivity_gradients"):
            predict_run_time(
                _shape("qwen3b"), RunShape(mixed=True), TuningConfig(), 4, gradient_source="x"
            )
        uniform = predict_run_time(
            _shape("qwen3b"), RunShape(), TuningConfig(), 4, gradient_source="x"
        )
        assert uniform.scoring_source is None

    def test_the_scoring_budget_is_the_planner_s(self) -> None:
        run = RunShape(mixed=True)
        prediction = predict_run_time(
            _shape("qwen0.5b"),
            run,
            TuningConfig(),
            4,
            scoring_samples=64,
            scoring_seq_len=512,
        )
        assert prediction.scoring_tokens == 64 * 512

    def test_the_steps_are_the_recipe_s(self) -> None:
        prediction = predict_run_time(_shape("qwen0.5b"), RunShape(), TuningConfig(iters=1000), 4)
        assert prediction.iters == 1000


class TestWhatItSays:
    """The line a run prints and the record it keeps."""

    def test_the_line_names_the_machine_and_the_band(self) -> None:
        prediction = predict_time(_shape("qwen3b"), RunShape(), iters=200, outlier_rule=False)
        line = prediction.describe()
        assert line.startswith("time: predicted ")
        assert MACHINE in line
        assert "36 blocks of" in line
        assert f"{SPREAD[0]:.2f} to {SPREAD[1]:.2f}" in line
        assert "scoring" not in line

    def test_a_mixed_run_names_its_scoring_pass(self) -> None:
        prediction = predict_time(
            _shape("qwen3b"), RunShape(mixed=True), iters=200, outlier_rule=False
        )
        assert "the scoring pass" in prediction.describe()

    def test_the_record_adds_up(self) -> None:
        prediction = predict_time(
            _shape("qwen7b"), RunShape(micro_batches=4), iters=200, outlier_rule=False
        )
        record = prediction.record()
        assert record["rule"] == RULE_VERSION
        assert record["machine"] == MACHINE
        assert record["seconds"] == pytest.approx(sum(record["by_stage_seconds"].values()), abs=0.2)
        assert record["blocks"] == 28
        assert record["pieces"] == 4
        assert record["spread"] == list(SPREAD)

    @pytest.mark.parametrize(
        ("seconds", "said"),
        [(45, "45s"), (434, "7m14s"), (3569, "59m29s"), (3570, "1h00m"), (16074, "4h28m")],
    )
    def test_a_span_reads_as_a_person_says_it(self, seconds: float, said: str) -> None:
        assert duration(seconds) == said
