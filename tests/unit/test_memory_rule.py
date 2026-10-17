# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The memory rule, held to every configuration it was fitted on and to what it promises.

``mround/planner/memory.py`` predicts a quantization run's peak from the
model's configuration and the run's settings, so that a run that would swap is
refused before it starts (ROAD-TO-1.0.md, M5). What it promises is the
configurations it reproduces, so those are written out here as the Mac
measured them, stage by stage at the last block: the five runs on the current
tree (``result.memory.by_block_bytes`` in each run's ``mround_run.json``) and
the eleven configurations of two synthetic blocks ``scripts/probe_memory.py``
measured on Metal (log0169). The rule is held to each: every peak and every
step between 1.5 percent short and 9 percent over, since its step was fitted
to err toward refusing, and a real run's peak within 1.2 percent; the passes
without a gradient within 6 percent either way; the measurements within a
percent on a whole batch and 7 in pieces; the scale search and the commit
never more than a percent short, nor more than 30 and 35 percent over; the
scoring pass to half a percent of its one measurement; and the entrance to a
few bytes, which is accounting. A refit that loses one fails here.

The rest is what the rule has to do whatever its constants: read a
configuration as the loader would, refuse to guess at one it cannot read, and
name the fewest pieces that fit.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import pytest

from mround.planner.memory import (
    BUDGET_SHARE,
    RULE_VERSION,
    ModelShape,
    RunShape,
    check_memory,
    default_budget,
    fitting_micro_batches,
    predict_memory,
)
from mround.schemes import QuantScheme, ScaleInit, Symmetry, TuningConfig

# The Mac every run below was measured on: 32 GB of unified memory.
MAC = 32 * 2**30

QWEN = {"model_type": "qwen2", "torch_dtype": "bfloat16"}
CONFIGS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": {
        **QWEN,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        **QWEN,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
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
QWEN_14B = {
    **QWEN,
    "hidden_size": 5120,
    "intermediate_size": 13824,
    "num_hidden_layers": 48,
    "num_attention_heads": 40,
    "num_key_value_heads": 8,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}

LOOP = ("targets", "search", "score", "tune", "commit", "advance")
SPLIT = ("32x512", "16x1024", "8x2048", "4x4096")


@dataclasses.dataclass(frozen=True)
class Measured:
    """One configuration as the Mac measured it: 2 bits, group size 64, searched scales.

    Attributes:
        source: The log, and for a run its directory.
        model: The shape measured.
        run: Its settings.
        last_block: Bytes each stage of the last block reached.
        peak: Bytes the run reached: its scoring pass on a mixed run, else the
            largest stage.
        entrance: Bytes the entrance reached, for a real run.
    """

    source: str
    model: ModelShape
    run: RunShape
    last_block: dict[str, int]
    peak: int
    entrance: int | None = None

    def __str__(self) -> str:
        """The test id: a run's log, or a probe configuration in full."""
        return self.source if self.entrance is None else self.source.split()[0]


def _config_shape(model: str) -> ModelShape:
    shape = ModelShape.from_config(CONFIGS[model])
    assert shape is not None
    return shape


def _recorded(
    log: str,
    directory: str,
    model: str,
    pieces: int,
    entrance: int,
    last_block: dict[str, int],
    peak: int,
    *,
    mixed: bool = False,
) -> Measured:
    # A mixed run's widest candidate was 4 bits, which only the export reads.
    run = RunShape(micro_batches=pieces, bits=4 if mixed else 2, mixed=mixed)
    return Measured(f"{log} {directory}", _config_shape(model), run, last_block, peak, entrance)


# Two blocks at a model's shape, a vocabulary of 1,024 and two batches, as
# scripts/probe_memory.py builds them; the second block is the one measured.
SMOLLM2 = (576, 1536, 9, 3)
QWEN_05B = (896, 4864, 14, 2)
QWEN_3B = (2048, 11008, 16, 2)
QWEN_7B = (3584, 18944, 28, 4)
LLAMA_8B = (4096, 14336, 32, 8)


def _probed(
    shape: tuple[int, int, int, int],
    batch: int,
    seq_len: int,
    gigabytes: dict[str, float],
    *,
    pieces: int = 1,
) -> Measured:
    hidden, intermediate, heads, kv_heads = shape
    model = ModelShape(
        hidden=hidden,
        intermediate=intermediate,
        layers=2,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=hidden // heads,
        vocab=1024,
    )
    run = RunShape(
        samples=2 * batch, seq_len=seq_len, batch_size=batch, micro_batches=pieces, bits=2
    )
    stages = {stage: round(value * 1e9) for stage, value in gigabytes.items()}
    name = f"log0169 {hidden}x{intermediate} {batch}x{seq_len} in {pieces}"
    return Measured(name, model, run, stages, max(stages.values()))


def _named(start: str) -> Measured:
    """The one configuration whose source starts so."""
    (found,) = [measured for measured in MEASURED if measured.source.startswith(start)]
    return found


# Every run on the tree since the losses were measured a batch at a time
# (MEMORY.md, M1 on Metal), at the standard recipe's 128 sequences of 2,048
# tokens in batches of 8, and every configuration of the probe of log0169,
# printed there to the megabyte.
MEASURED = [
    _recorded(
        "log0152",
        "parity-qwen2.5-1.5b-instruct-2bit-g64-denserem-mb8-m1",
        "Qwen/Qwen2.5-1.5B-Instruct",
        8,
        2_883_715_080,
        {
            "targets": 7_343_593_866,
            "search": 8_183_251_414,
            "score": 7_129_100_710,
            "tune": 8_263_006_710,
            "commit": 6_217_888_202,
            "advance": 7_393_352_074,
        },
        8_263_006_710,
    ),
    _recorded(
        "log0157",
        "parity-qwen2.5-3b-instruct-2bit-g64-denserem-tree1012",
        "Qwen/Qwen2.5-3B-Instruct",
        1,
        3_844_603_912,
        {
            "targets": 10_587_739_642,
            "search": 11_755_459_142,
            "score": 12_430_774_846,
            "tune": 20_069_954_386,
            "commit": 9_387_364_886,
            "advance": 10_669_626_874,
        },
        20_069_954_386,
    ),
    _recorded(
        "log0156",
        "parity-qwen2.5-3b-instruct-2bit-g64-denserem-memprobe-l1",
        "Qwen/Qwen2.5-3B-Instruct",
        1,
        3_844_603_912,
        {
            "targets": 10_587_739_642,
            "search": 11_755_459_142,
            "score": 12_430_774_846,
            "tune": 20_069_954_386,
            "commit": 9_705_280_022,
            "advance": 10_669_626_874,
        },
        20_069_954_386,
    ),
    _recorded(
        "log0160",
        "parity-qwen2.5-3b-instruct-avg2.5bit-o2-3-4-g64-widest-denserem-memprobe-l2",
        "Qwen/Qwen2.5-3B-Instruct",
        1,
        3_844_603_912,
        {
            "targets": 10_587_739_642,
            "search": 11_755_459_142,
            "score": 11_596_755_514,
            "tune": 20_069_954_386,
            "commit": 9_705_280_022,
            "advance": 10_669_626_874,
        },
        23_948_439_208,
        mixed=True,
    ),
    _recorded(
        "log0155",
        "parity-qwen2.5-7b-instruct-2bit-g64-denserem-mb8-memprobe",
        "Qwen/Qwen2.5-7B-Instruct",
        8,
        6_728_187_912,
        {
            "targets": 20_640_647_562,
            "search": 22_766_619_094,
            "score": 23_942_912_510,
            "tune": 26_284_167_666,
            "commit": 21_022_351_778,
            "advance": 20_888_258_954,
        },
        26_284_167_666,
    ),
    _probed(
        QWEN_7B,
        8,
        2048,
        {
            "targets": 6.542,
            "search": 8.668,
            "score": 11.179,
            "tune": 24.75,
            "commit": 6.924,
            "advance": 6.79,
        },
    ),
    _probed(
        QWEN_7B,
        8,
        2048,
        {
            "targets": 6.542,
            "search": 8.668,
            "score": 9.416,
            "tune": 18.18,
            "commit": 6.924,
            "advance": 6.79,
        },
        pieces=2,
    ),
    _probed(
        QWEN_7B,
        8,
        2048,
        {
            "targets": 6.542,
            "search": 8.668,
            "score": 9.416,
            "tune": 14.744,
            "commit": 6.924,
            "advance": 6.79,
        },
        pieces=4,
    ),
    _probed(
        QWEN_7B,
        8,
        2048,
        {
            "targets": 6.542,
            "search": 8.668,
            "score": 9.609,
            "tune": 12.185,
            "commit": 6.924,
            "advance": 6.79,
        },
        pieces=8,
    ),
    _probed(
        SMOLLM2,
        8,
        2048,
        {
            "targets": 0.718,
            "search": 1.009,
            "score": 1.184,
            "tune": 5.223,
            "commit": 0.26,
            "advance": 0.722,
        },
    ),
    _probed(
        QWEN_05B,
        8,
        2048,
        {
            "targets": 1.475,
            "search": 2.087,
            "score": 2.113,
            "tune": 8.664,
            "commit": 0.727,
            "advance": 1.491,
        },
    ),
    _probed(
        LLAMA_8B,
        8,
        2048,
        {
            "targets": 6.33,
            "search": 8.127,
            "score": 10.607,
            "tune": 25.89,
            "commit": 6.506,
            "advance": 6.562,
        },
    ),
    _probed(
        QWEN_3B,
        32,
        512,
        {
            "targets": 3.426,
            "search": 4.594,
            "score": 5.269,
            "tune": 10.646,
            "commit": 2.544,
            "advance": 3.508,
        },
    ),
    _probed(
        QWEN_3B,
        16,
        1024,
        {
            "targets": 3.426,
            "search": 4.594,
            "score": 5.269,
            "tune": 11.184,
            "commit": 2.544,
            "advance": 3.508,
        },
    ),
    _probed(
        QWEN_3B,
        8,
        2048,
        {
            "targets": 3.426,
            "search": 4.594,
            "score": 5.269,
            "tune": 12.908,
            "commit": 2.544,
            "advance": 3.508,
        },
    ),
    _probed(
        QWEN_3B,
        4,
        4096,
        {
            "targets": 3.426,
            "search": 4.594,
            "score": 5.269,
            "tune": 19.363,
            "commit": 2.544,
            "advance": 3.508,
        },
    ),
]
RECORDS = [measured for measured in MEASURED if measured.entrance is not None]
WHOLE = [m for m in MEASURED if not m.run.mixed and m.run.micro_batches == 1]
PIECES = [m for m in MEASURED if m.run.micro_batches > 1]


def _shape(model: str) -> ModelShape:
    return _config_shape(model)


def _error(predicted: int, measured: int) -> float:
    return (predicted - measured) / measured


def _within(predicted: int, measured: int, short: float, over: float) -> bool:
    return -short < _error(predicted, measured) < over


class TestEveryMeasurement:
    """Every configuration the rule was fitted on, reproduced within its bounds."""

    @pytest.mark.parametrize("measured", MEASURED, ids=str)
    def test_the_peak_errs_toward_refusing(self, measured: Measured) -> None:
        prediction = predict_memory(measured.model, measured.run)
        stages = (*LOOP, "scoring") if measured.run.mixed else LOOP
        peak = max(prediction.stages[stage] for stage in stages)
        assert _within(peak, measured.peak, 0.015, 0.09)

    @pytest.mark.parametrize("measured", MEASURED, ids=str)
    def test_the_step_errs_toward_refusing(self, measured: Measured) -> None:
        prediction = predict_memory(measured.model, measured.run)
        assert _within(prediction.stages["tune"], measured.last_block["tune"], 0.015, 0.09)

    @pytest.mark.parametrize("measured", MEASURED, ids=str)
    def test_the_stage_that_sets_the_peak_is_the_one_that_did(self, measured: Measured) -> None:
        prediction = predict_memory(measured.model, measured.run)
        stages = (*LOOP, "scoring") if measured.run.mixed else LOOP
        expected = "scoring" if measured.run.mixed else "tune"
        assert max(stages, key=prediction.stages.__getitem__) == expected

    @pytest.mark.parametrize("measured", MEASURED, ids=str)
    def test_the_passes_without_a_gradient_are_within_six_percent(self, measured: Measured) -> None:
        prediction = predict_memory(measured.model, measured.run)
        for stage in ("targets", "advance"):
            assert _within(prediction.stages[stage], measured.last_block[stage], 0.06, 0.06)

    @pytest.mark.parametrize("measured", WHOLE, ids=str)
    def test_the_measurements_of_a_whole_batch_are_within_a_percent(
        self, measured: Measured
    ) -> None:
        # A mixed block's measurements carry the widths it was given, which the
        # uniform form does not count, so the mixed run sits out (7.2 percent
        # over); its steps are a uniform run's, and are held above.
        prediction = predict_memory(measured.model, measured.run)
        assert _within(prediction.stages["score"], measured.last_block["score"], 0.01, 0.01)

    @pytest.mark.parametrize("measured", PIECES, ids=str)
    def test_the_measurements_in_pieces_are_within_seven_percent(self, measured: Measured) -> None:
        # Fitted on whole batches; in pieces they fall between 5.5 percent
        # short and 6 percent over, and are never the peak.
        prediction = predict_memory(measured.model, measured.run)
        assert _within(prediction.stages["score"], measured.last_block["score"], 0.07, 0.07)

    @pytest.mark.parametrize("measured", RECORDS, ids=str)
    def test_a_real_run_s_peak_is_within_a_percent_and_a_fifth(self, measured: Measured) -> None:
        # Tighter than the probe's bound: these are the runs a user makes.
        prediction = predict_memory(measured.model, measured.run)
        assert _within(prediction.peak, measured.peak, 0.012, 0.012)

    def test_the_scoring_pass_reproduces_its_one_measurement(self) -> None:
        # A 3B at the matched budget, the one scoring pass on the current tree
        # (log0160), on which the logits' share is fitted.
        mixed = _named("log0160")
        prediction = predict_memory(mixed.model, mixed.run)
        assert _within(prediction.stages["scoring"], mixed.peak, 0.005, 0.005)

    @pytest.mark.parametrize("measured", MEASURED, ids=str)
    def test_the_search_and_the_commit_err_toward_refusing(self, measured: Measured) -> None:
        # Each carries a fixed term sized for the largest shapes, so the
        # smallest, SmolLM2-135M, sits furthest over: 25 and 33 percent.
        prediction = predict_memory(measured.model, measured.run)
        assert _within(prediction.stages["search"], measured.last_block["search"], 0.01, 0.30)
        assert _within(prediction.stages["commit"], measured.last_block["commit"], 0.01, 0.35)

    @pytest.mark.parametrize("measured", RECORDS, ids=str)
    def test_the_entrance_is_accounting_to_a_few_bytes(self, measured: Measured) -> None:
        # The reference stream, the captured inputs, the embedding and the
        # tokens; eight bytes short on every run.
        prediction = predict_memory(measured.model, measured.run)
        assert measured.entrance is not None
        assert abs(prediction.stages["entrance"] - measured.entrance) < 64

    @pytest.mark.parametrize("measured", RECORDS, ids=str)
    def test_after_the_loop_stays_under_the_loop(self, measured: Measured) -> None:
        # Estimated by accounting, not yet measured: the claim the module makes
        # about them is that they do not set the peak on these runs.
        prediction = predict_memory(measured.model, measured.run)
        loop = max(prediction.stages[stage] for stage in LOOP)
        assert prediction.stages["remainder"] < loop
        assert prediction.stages["export"] < loop

    @pytest.mark.parametrize(
        ("probed", "recorded"),
        [
            ("log0169 2048x11008 8x2048 in 1", "log0156"),
            ("log0169 3584x18944 8x2048 in 8", "log0155"),
        ],
        ids=["3B whole", "7B in eight pieces"],
    )
    def test_the_probe_holds_what_a_real_run_holds_above_the_accounting(
        self, probed: str, recorded: str
    ) -> None:
        # Two synthetic blocks against the last block of a real run, each with
        # what the accounting gives subtracted: the same working set, to the
        # megabyte, in every stage but the 7B's measurements in pieces, where
        # the run's sixteen batches held 0.24 GB more than the probe's two.
        probe, run = _named(probed), _named(recorded)
        probe_floor = sum(predict_memory(probe.model, probe.run).parts.values())
        run_floor = sum(predict_memory(run.model, run.run).parts.values())
        for stage in LOOP:
            if stage == "score" and probe.run.micro_batches > 1:
                continue
            above_probe = probe.last_block[stage] - probe_floor
            above_run = run.last_block[stage] - run_floor
            assert abs(above_probe - above_run) < 2e6, stage

    def test_the_block_weights_are_the_layers_the_loop_tunes(self) -> None:
        # Seven linear layers a block: q, k, v and o, and a gated MLP's three.
        # The resident accounting the rule rests on matched every block of
        # every recorded run with these.
        assert _shape("Qwen/Qwen2.5-1.5B-Instruct").block_weights == 46_792_704
        assert _shape("Qwen/Qwen2.5-3B-Instruct").block_weights == 77_070_336
        assert _shape("Qwen/Qwen2.5-7B-Instruct").block_weights == 233_046_016

    def test_the_step_grows_faster_than_the_sequence_at_a_fixed_number_of_tokens(
        self,
    ) -> None:
        # The split suite, which no sum of an MLP term and an attention term
        # fits: at 16,384 tokens a step, each doubling of the sequence added
        # 0.54, 1.72 and then 6.46 GB. The rule's step grows faster at each
        # doubling too, as the larger of the two terms does.
        split = [_named(f"log0169 2048x11008 {shape}") for shape in SPLIT]
        grew = [
            predict_memory(after.model, after.run).stages["tune"]
            - predict_memory(before.model, before.run).stages["tune"]
            for before, after in itertools.pairwise(split)
        ]
        measured = [
            after.last_block["tune"] - before.last_block["tune"]
            for before, after in itertools.pairwise(split)
        ]
        assert measured == sorted(measured)
        assert grew == sorted(grew)
        assert grew[-1] > 2 * grew[-2]


class TestWhatTheRuleSays:
    """The decisions the rule makes on the Mac the runs were measured on."""

    def test_a_7b_fits_from_four_pieces(self) -> None:
        check = check_memory(_shape("Qwen/Qwen2.5-7B-Instruct"), RunShape(bits=2), device=MAC)
        assert not check.fits
        assert check.micro_batches == 4
        assert "--micro-batches 4" in check.explain()
        assert "micro_batches=4" in check.explain()

    def test_a_3b_fits_whole(self) -> None:
        check = check_memory(_shape("Qwen/Qwen2.5-3B-Instruct"), RunShape(bits=2), device=MAC)
        assert check.fits
        assert check.micro_batches is None
        assert check.explain() == ""
        assert "within the budget" in check.describe()

    def test_pieces_cannot_fit_a_7b_mixed_run(self) -> None:
        run = RunShape(bits=4, mixed=True)
        check = check_memory(_shape("Qwen/Qwen2.5-7B-Instruct"), run, device=MAC)
        assert check.prediction.stage == "scoring"
        assert check.micro_batches is None
        text = check.explain()
        assert "scoring pass" in text
        # The uniform run is the alternative, and it needs its pieces too.
        assert "A uniform run does not score" in text
        assert "--micro-batches 4" in text

    def test_nothing_fits_a_14b_and_the_message_says_what_holds_it(self) -> None:
        shape = ModelShape.from_config(QWEN_14B)
        assert shape is not None
        check = check_memory(shape, RunShape(bits=2), device=MAC)
        assert check.micro_batches is None
        text = check.explain()
        assert "No number of pieces fits" in text
        assert "--samples" in text
        assert "--seq-len" in text

    def test_every_refusal_says_how_to_run_anyway(self) -> None:
        check = check_memory(_shape("Qwen/Qwen2.5-7B-Instruct"), RunShape(bits=2), device=MAC)
        assert "--memory-budget" in check.explain()
        assert "--skip-memory-check" in check.explain()


class TestTheBudget:
    """What a prediction is held to."""

    def test_the_default_is_a_share_of_the_device(self) -> None:
        assert default_budget(MAC) == int(MAC * BUDGET_SHARE)
        assert default_budget(0) is None

    def test_an_unknown_device_checks_nothing(self) -> None:
        check = check_memory(_shape("Qwen/Qwen2.5-7B-Instruct"), RunShape(bits=2), device=0)
        assert check.budget is None
        assert check.fits
        assert "unknown" in check.describe()

    def test_a_budget_passed_wins_over_the_device(self) -> None:
        shape = _shape("Qwen/Qwen2.5-3B-Instruct")
        check = check_memory(shape, RunShape(bits=2), budget=15 * 10**9, device=MAC)
        assert check.caller_budget
        assert not check.fits
        assert check.micro_batches is not None
        fitted = predict_memory(shape, RunShape(bits=2, micro_batches=check.micro_batches))
        assert fitted.peak <= 15 * 10**9
        assert "it was given" in check.describe()

    def test_a_budget_passed_works_without_a_device(self) -> None:
        check = check_memory(
            _shape("Qwen/Qwen2.5-7B-Instruct"), RunShape(bits=2), budget=30 * 10**9
        )
        assert not check.fits
        assert check.micro_batches == 4

    @pytest.mark.parametrize("budget", [0, -1])
    def test_a_budget_that_is_not_positive_is_refused(self, budget: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            check_memory(_shape("Qwen/Qwen2.5-3B-Instruct"), RunShape(), budget=budget)

    def test_the_record_says_what_was_held_to_what(self) -> None:
        check = check_memory(_shape("Qwen/Qwen2.5-7B-Instruct"), RunShape(bits=2), device=MAC)
        record = check.record(enforced=True)
        assert record["rule"] == RULE_VERSION
        assert record["peak_bytes"] == check.prediction.peak
        assert record["stage"] == "tune"
        assert record["budget_bytes"] == int(MAC * BUDGET_SHARE)
        assert record["budget_source"] == f"{BUDGET_SHARE:g} of the device"
        assert record["device_bytes"] == MAC
        assert record["fits"] is False
        assert record["enforced"] is True
        assert record["fitting_micro_batches"] == 4
        assert record["model"]["hidden"] == 3584
        assert record["run"]["micro_batches"] == 1

    def test_a_partial_run_is_held_to_its_own_stages(self) -> None:
        # The planning pass alone runs only the scoring pass.
        shape = _shape("Qwen/Qwen2.5-3B-Instruct")
        check = check_memory(shape, RunShape(mixed=True), device=MAC, stages={"scoring"})
        assert set(check.prediction.stages) == {"scoring"}
        assert check.only == frozenset({"scoring"})
        tight = check_memory(
            shape, RunShape(mixed=True), budget=10**9, device=MAC, stages={"scoring"}
        )
        assert tight.micro_batches is None
        assert "scoring pass" in tight.explain()
        assert "uniform" not in tight.explain()


class TestPieces:
    """The fewest pieces that fit, and what pieces change."""

    def test_the_fewest_is_returned(self) -> None:
        shape = _shape("Qwen/Qwen2.5-7B-Instruct")
        run = RunShape(bits=2)
        pieces = fitting_micro_batches(shape, run, int(MAC * BUDGET_SHARE))
        assert pieces == 4
        assert predict_memory(shape, run.with_micro_batches(3)).peak > MAC * BUDGET_SHARE

    def test_a_run_that_fits_needs_one(self) -> None:
        shape = _shape("Qwen/Qwen2.5-3B-Instruct")
        assert fitting_micro_batches(shape, RunShape(bits=2), MAC) == 1

    def test_none_when_nothing_fits(self) -> None:
        shape = _shape("Qwen/Qwen2.5-7B-Instruct")
        assert fitting_micro_batches(shape, RunShape(bits=2), 10**9) is None

    @pytest.mark.parametrize("model", sorted(CONFIGS))
    def test_more_pieces_never_need_more(self, model: str) -> None:
        shape = _shape(model)
        peaks = [predict_memory(shape, RunShape(micro_batches=m)).peak for m in range(1, 9)]
        assert peaks == sorted(peaks, reverse=True)

    def test_pieces_change_only_the_steps_and_the_measurements(self) -> None:
        shape = _shape("Qwen/Qwen2.5-7B-Instruct")
        whole = predict_memory(shape, RunShape()).stages
        split = predict_memory(shape, RunShape(micro_batches=8)).stages
        for stage in ("entrance", "targets", "search", "commit", "advance", "export"):
            assert whole[stage] == split[stage], stage
        assert split["tune"] < whole["tune"]


class TestRunSettings:
    """What each setting moves."""

    @pytest.mark.parametrize("model", sorted(CONFIGS))
    def test_more_samples_hold_more_in_every_loop_stage(self, model: str) -> None:
        shape = _shape(model)
        fewer = predict_memory(shape, RunShape(samples=64)).stages
        more = predict_memory(shape, RunShape(samples=128)).stages
        for stage in ("entrance", "targets", "search", "score", "tune", "commit", "advance"):
            assert more[stage] > fewer[stage], stage
        # After the loop the streams are gone, so the samples change nothing.
        assert more["remainder"] == fewer["remainder"]

    def test_the_streams_are_two_float32_copies_of_every_token(self) -> None:
        shape = _shape("Qwen/Qwen2.5-3B-Instruct")
        prediction = predict_memory(shape, RunShape())
        assert prediction.parts["streams"] == 2 * 128 * 2048 * 2048 * 4

    @pytest.mark.parametrize("model", sorted(CONFIGS))
    def test_every_loop_stage_grows_with_the_samples_by_the_streams_and_tokens(
        self, model: str
    ) -> None:
        # Sixty four more sequences of 2,048 tokens: two float32 streams of
        # them, and the tokens themselves at four bytes each, in every stage
        # that holds the streams and nowhere else.
        shape = _shape(model)
        fewer = predict_memory(shape, RunShape(samples=64)).stages
        more = predict_memory(shape, RunShape(samples=128)).stages
        tokens = 64 * 2048
        for stage in ("targets", "search", "score", "tune", "commit", "advance"):
            assert more[stage] - fewer[stage] == pytest.approx(
                2 * tokens * shape.hidden * 4 + 4 * tokens, abs=2
            ), stage

    def test_the_stages_after_the_loop_are_accounting(self) -> None:
        # Every block's codes, the tensors outside the blocks at the model's
        # width, and four float32 copies of the embedding for its rounding;
        # or the packed checkpoint at 2 bits with a scale and a bias a group.
        shape = _shape("Qwen/Qwen2.5-7B-Instruct")
        stages = predict_memory(shape, RunShape(bits=2)).stages
        codes = shape.layers * shape.block_weights * (1 + 4 / 64)
        outside = 2 * shape.embedding_weights * 2
        assert stages["remainder"] == round(codes + outside + 16 * shape.embedding_weights)
        packed = shape.layers * shape.block_weights * (2 / 8 + 2 * 2 / 64)
        assert stages["export"] == round(codes + outside + packed)

    def test_a_zero_point_costs_four_bytes_a_group(self) -> None:
        shape = _shape("Qwen/Qwen2.5-3B-Instruct")
        symmetric = predict_memory(shape, RunShape()).parts["codes"]
        asymmetric = predict_memory(shape, RunShape(asymmetric=True)).parts["codes"]
        groups = (shape.layers - 1) * shape.block_weights // 64
        assert asymmetric - symmetric == pytest.approx(4 * groups, rel=1e-6)

    def test_without_the_search_there_is_no_search_working_set(self) -> None:
        shape = _shape("Qwen/Qwen2.5-3B-Instruct")
        searched = predict_memory(shape, RunShape(searched=True)).stages["search"]
        observed = predict_memory(shape, RunShape(searched=False)).stages["search"]
        assert observed < searched

    def test_a_mixed_run_adds_its_scoring_pass(self) -> None:
        shape = _shape("Qwen/Qwen2.5-3B-Instruct")
        assert "scoring" not in predict_memory(shape, RunShape()).stages
        mixed = predict_memory(shape, RunShape(mixed=True)).stages
        # One float32 copy of every block weight, beside the whole model.
        assert mixed["scoring"] > 4 * shape.layers * shape.block_weights


class TestReadingAConfiguration:
    """What the rule reads, and what it declines to guess at."""

    def test_a_multimodal_configuration_is_read_through_its_text_model(self) -> None:
        nested = {"model_type": "gemma3", "text_config": CONFIGS["Qwen/Qwen2.5-3B-Instruct"]}
        assert ModelShape.from_config(nested) == _shape("Qwen/Qwen2.5-3B-Instruct")

    @pytest.mark.parametrize("key", ["num_local_experts", "num_experts", "n_routed_experts"])
    def test_a_mixture_of_experts_gets_no_shape(self, key: str) -> None:
        config = {**CONFIGS["Qwen/Qwen2.5-3B-Instruct"], key: 8}
        assert ModelShape.from_config(config) is None

    @pytest.mark.parametrize(
        "change",
        [
            {"hidden_size": None},
            {"num_attention_heads": 0},
            {"vocab_size": "many"},
        ],
    )
    def test_a_configuration_it_cannot_read_gets_no_shape(self, change: dict[str, Any]) -> None:
        config = {**CONFIGS["Qwen/Qwen2.5-3B-Instruct"], **change}
        assert ModelShape.from_config(config) is None

    def test_a_missing_size_gets_no_shape(self) -> None:
        config = dict(CONFIGS["Qwen/Qwen2.5-3B-Instruct"])
        del config["intermediate_size"]
        assert ModelShape.from_config(config) is None

    def test_an_explicit_head_width_is_used(self) -> None:
        # Gemma's heads are wider than the hidden width divided among them.
        config = {**CONFIGS["Qwen/Qwen2.5-3B-Instruct"], "head_dim": 256}
        shape = ModelShape.from_config(config)
        assert shape is not None
        assert shape.head_dim == 256
        assert shape.block_weights > _shape("Qwen/Qwen2.5-3B-Instruct").block_weights

    def test_without_key_value_heads_every_head_has_its_own(self) -> None:
        config = dict(CONFIGS["Qwen/Qwen2.5-3B-Instruct"])
        del config["num_key_value_heads"]
        shape = ModelShape.from_config(config)
        assert shape is not None
        assert shape.kv_heads == shape.heads

    def test_an_unstated_tie_is_read_as_untied(self) -> None:
        # One more matrix to hold, so a wrong guess errs toward refusing.
        config = dict(CONFIGS["Qwen/Qwen2.5-3B-Instruct"])
        del config["tie_word_embeddings"]
        shape = ModelShape.from_config(config)
        assert shape is not None
        assert not shape.tied
        assert shape.outside_weights == 2 * shape.embedding_weights

    @pytest.mark.parametrize(
        ("dtype", "width"),
        [("bfloat16", 2), ("float16", 2), ("torch.float16", 2), ("float32", 4), ("int8", 2)],
    )
    def test_the_width_comes_from_the_dtype(self, dtype: str, width: int) -> None:
        config = {**CONFIGS["Qwen/Qwen2.5-3B-Instruct"], "torch_dtype": dtype}
        shape = ModelShape.from_config(config)
        assert shape is not None
        assert shape.dtype_bytes == width

    def test_the_newer_dtype_key_is_read_too(self) -> None:
        config = dict(CONFIGS["Qwen/Qwen2.5-3B-Instruct"])
        del config["torch_dtype"]
        config["dtype"] = "float32"
        shape = ModelShape.from_config(config)
        assert shape is not None
        assert shape.dtype_bytes == 4

    def test_an_ungated_mlp_has_two_matrices(self) -> None:
        config = {**CONFIGS["Qwen/Qwen2.5-3B-Instruct"], "model_type": "phi"}
        shape = ModelShape.from_config(config)
        assert shape is not None
        assert not shape.gated
        gated = _shape("Qwen/Qwen2.5-3B-Instruct")
        assert gated.block_weights - shape.block_weights == shape.hidden * shape.intermediate


class TestRunShapeFromSettings:
    """A run's shape resolves the way the run does."""

    def test_the_standard_recipe(self) -> None:
        scheme = QuantScheme(bits=2, group_size=64, scale_init=ScaleInit.SEARCHED)
        run = RunShape.of(scheme, TuningConfig())
        assert run == RunShape(bits=2, searched=True, mixed=False, candidate_widths=1)

    def test_pieces_and_the_recipe_come_from_the_tuning(self) -> None:
        tuning = TuningConfig(n_samples=64, seq_len=1024, batch_size=4, micro_batches=2)
        run = RunShape.of(QuantScheme(bits=4), tuning)
        assert (run.samples, run.seq_len, run.batch_size, run.micro_batches) == (64, 1024, 4, 2)
        assert not run.searched

    def test_a_mixed_run_scores_its_candidates_in_batches_the_budget_derives(self) -> None:
        run = RunShape.of(QuantScheme(bits=4), TuningConfig(), candidate_bits=(2, 3, 4, 3))
        assert run.mixed
        assert run.candidate_widths == 3
        assert run.bits == 4
        # 2,048 scoring tokens a batch at 1,024 a sequence.
        assert (run.scoring_sequences, run.scoring_seq_len) == (2, 1024)

    def test_the_scoring_settings_are_honored(self) -> None:
        run = RunShape.of(
            QuantScheme(bits=4),
            TuningConfig(),
            candidate_bits=(2, 4),
            scoring_samples=16,
            scoring_seq_len=256,
            scoring_batch_size=0,
        )
        assert (run.scoring_sequences, run.scoring_seq_len) == (1, 256)

    def test_the_widest_candidate_sizes_the_packed_codes(self) -> None:
        run = RunShape.of(QuantScheme(bits=2), TuningConfig(), candidate_bits=(2, 8))
        assert run.bits == 8

    def test_an_asymmetric_scheme_is_seen(self) -> None:
        run = RunShape.of(QuantScheme(bits=4, symmetry=Symmetry.ASYMMETRIC), TuningConfig())
        assert run.asymmetric
