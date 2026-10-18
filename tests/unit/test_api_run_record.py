# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""What ``api.quantize`` and its block runner record and refuse.

Run on a tiny model built in memory, so there is no download and no Apple
hardware: the host check and the loader are replaced, and the model is an
``mlx-lm`` Llama with two blocks of width 64.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("mlx.core", reason="the block loop is the MLX layer")
pytest.importorskip("mlx_lm", reason="the tiny model is an mlx-lm architecture")

import mlx.core as mx
from mlx_lm.models import llama

from mround import api
from mround.exceptions import ArchitectureError, CalibrationError, MemoryBudgetError, SchemeError
from mround.pipeline import loader
from mround.pipeline.runner import QuantizationRunner
from mround.planner.memory import RULE_VERSION, ModelShape, RunShape, predict_memory
from mround.schemes import QuantScheme, TuningConfig

if TYPE_CHECKING:
    from pathlib import Path

CONFIG: dict[str, Any] = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": 64,
    "tie_word_embeddings": True,
}
TUNING = TuningConfig(iters=2, n_samples=1, seq_len=16, batch_size=1)


class _Tokenizer:
    bos_token_id = None
    eos_token_id = 1

    def encode(self, text: str, **_: Any) -> list[int]:
        return [2 + (ord(c) % 60) for c in text]

    def save_pretrained(self, _: Any) -> None:
        return None


def _bundle() -> loader.ModelBundle:
    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs(**CONFIG))
    mx.eval(model.parameters())
    return loader.ModelBundle(
        model=model, tokenizer=_Tokenizer(), config=dict(CONFIG), source="tiny", dtype=mx.float32
    )


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(api, "_require_supported_host", lambda: None)
    monkeypatch.setattr(loader, "load_model", lambda *_, **__: _bundle())
    # The memory check reads the configuration before anything loads, which
    # for a name that is not a directory means the Hub; this model has none.
    monkeypatch.setattr(loader, "read_config", lambda *_: dict(CONFIG))
    path = tmp_path / "corpus.txt"
    path.write_text("the quick brown fox jumps over the lazy dog " * 40, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("quantized_inputs", [True, False])
def test_the_run_record_names_the_activation_stream(
    corpus: str, tmp_path: Path, quantized_inputs: bool
) -> None:
    # Which stream every block was tuned against changes the checkpoint and
    # cannot be read back from it, so the file that claims to reproduce the
    # run has to carry it (Q-001).
    result = api.quantize(
        "tiny",
        output_dir=tmp_path / "out",
        tuning=TUNING,
        calibration=corpus,
        quantized_inputs=quantized_inputs,
    )
    record = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert record["quantized_inputs"] is quantized_inputs


def test_a_negative_holdout_is_refused_before_the_model_loads(
    corpus: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unrefused, it loaded the model, built the corpus and then divided by
    # zero inside the first block.
    def must_not_load(*_: Any, **__: Any) -> None:
        msg = "the model was loaded for a run that could never tune"
        raise AssertionError(msg)

    monkeypatch.setattr(loader, "load_model", must_not_load)
    with pytest.raises(CalibrationError, match="holdout_batches"):
        api.quantize(
            "tiny",
            output_dir=tmp_path / "out",
            tuning=TUNING,
            calibration=corpus,
            holdout_batches=-1,
        )


def test_the_runner_refuses_a_negative_holdout() -> None:
    with pytest.raises(CalibrationError, match="holdout_batches"):
        QuantizationRunner(QuantScheme(), TUNING, holdout_batches=-1)
    # Zero and a holdout larger than the corpus stay accepted; the second
    # degrades to tuning on one batch, as the runner documents.
    QuantizationRunner(QuantScheme(), TUNING, holdout_batches=0)
    QuantizationRunner(QuantScheme(), TUNING, holdout_batches=100)


@pytest.mark.parametrize("batch", [0, -3])
def test_a_scoring_batch_below_one_is_refused(corpus: str, batch: int) -> None:
    # It was clamped to 1 without a word, which is a run other than the one
    # asked for.
    with pytest.raises(SchemeError, match="scoring_batch_size"):
        api.plan_mixed_precision(
            "tiny",
            3.0,
            (2, 4),
            scoring_samples=1,
            scoring_seq_len=16,
            scoring_batch_size=batch,
            calibration=corpus,
        )


def test_a_string_symmetry_resolves_the_same_initialization(corpus: str, tmp_path: Path) -> None:
    # QuantScheme coerces "sym", so the run is symmetric either way; the
    # initialization it gets at 2 bits must not depend on the spelling.
    result = api.quantize(
        "tiny",
        output_dir=tmp_path / "out",
        bits=2,
        symmetry="sym",  # type: ignore[arg-type]
        tuning=TUNING,
        calibration=corpus,
    )
    record = json.loads(result.config_path.read_text(encoding="utf-8"))
    assert record["scheme"]["scale_init"] == "searched"


def test_a_mixed_run_tunes_a_fresh_lazy_load_not_the_scored_model(
    corpus: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The scoring backward reads every weight, and the scored model kept them
    # resident through the block loop: 5.4 GB at a 3B model's first block
    # (log0151). The block loop is given a second lazy load instead.
    loads: list[tuple[Any, dict[str, Any]]] = []

    def load(*_: Any, **kwargs: Any) -> loader.ModelBundle:
        bundle = _bundle()
        loads.append((bundle.model, kwargs))
        return bundle

    tuned: list[Any] = []
    run = QuantizationRunner.run

    def recording_run(self: QuantizationRunner, model: Any, *args: Any, **kwargs: Any) -> Any:
        tuned.append(model)
        return run(self, model, *args, **kwargs)

    monkeypatch.setattr(loader, "load_model", load)
    monkeypatch.setattr(QuantizationRunner, "run", recording_run)
    api.quantize(
        "tiny",
        output_dir=tmp_path / "out",
        average_bits=3.0,
        candidate_bits=(2, 4),
        scoring_samples=1,
        scoring_seq_len=16,
        tuning=TUNING,
        calibration=corpus,
    )
    assert [kwargs for _, kwargs in loads] == [{"lazy": True}, {"lazy": True}]
    assert len(tuned) == 1
    assert tuned[0] is loads[1][0]
    assert tuned[0] is not loads[0][0]


def _must_not_load(*_: Any, **__: Any) -> None:
    msg = "the model was loaded for a run the memory check should have refused"
    raise AssertionError(msg)


class TestTheMemoryCheck:
    """The prediction before a run, and the measurement after its loop (M5)."""

    def test_the_run_record_carries_the_prediction_beside_the_measurement(
        self, corpus: str, tmp_path: Path
    ) -> None:
        result = api.quantize(
            "tiny", output_dir=tmp_path / "out", bits=2, tuning=TUNING, calibration=corpus
        )
        record = json.loads(result.config_path.read_text(encoding="utf-8"))
        predicted = record["memory_check"]
        assert predicted["rule"] == RULE_VERSION
        assert predicted["enforced"] is True
        assert predicted["fits"] is True
        assert set(predicted["by_stage_bytes"]) >= {"entrance", "tune", "remainder", "export"}
        assert predicted["model"]["layers"] == CONFIG["num_hidden_layers"]
        assert predicted["run"]["samples"] == TUNING.n_samples

    def test_the_stages_after_the_loop_are_measured(self, corpus: str, tmp_path: Path) -> None:
        result = api.quantize(
            "tiny", output_dir=tmp_path / "out", tuning=TUNING, calibration=corpus
        )
        memory = json.loads(result.config_path.read_text(encoding="utf-8"))["result"]["memory"]
        assert {"remainder", "export"} <= set(memory["by_stage_bytes"])
        assert set(result.summary.after_blocks) == {"remainder", "export"}
        # The run's peak is the whole run's now, the export included.
        assert memory["peak_bytes"] >= max(memory["by_stage_bytes"].values())

    def test_a_dense_remainder_is_left_dense_and_predicted_as_one(
        self, corpus: str, tmp_path: Path
    ) -> None:
        # Left dense, the remainder rounds nothing, the checkpoint keeps the
        # embedding at the model's width, and the check says so.
        result = api.quantize(
            "tiny",
            output_dir=tmp_path / "out",
            tuning=TUNING,
            calibration=corpus,
            dense_remainder=True,
        )
        record = json.loads(result.config_path.read_text(encoding="utf-8"))
        assert [
            "model.embed_tokens",
            "dense by request, to match a reference's coverage",
        ] in record["result"]["skipped"]
        predicted = record["memory_check"]
        assert predicted["run"]["dense_remainder"] is True
        shape = ModelShape(**predicted["model"])
        rounded = RunShape(**{**predicted["run"], "dense_remainder": False})
        remainder = predicted["by_stage_bytes"]["remainder"]
        assert remainder < predict_memory(shape, rounded).stages["remainder"]

    def test_a_run_past_its_budget_is_refused_before_the_model_loads(
        self, corpus: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(loader, "load_model", _must_not_load)
        with pytest.raises(MemoryBudgetError, match="predicted to reach") as caught:
            api.quantize(
                "tiny",
                output_dir=tmp_path / "out",
                tuning=TUNING,
                calibration=corpus,
                memory_budget=1,
            )
        assert "--memory-budget" in str(caught.value)
        assert not (tmp_path / "out").exists()

    def test_with_the_check_off_the_run_is_recorded_and_made(
        self, corpus: str, tmp_path: Path
    ) -> None:
        result = api.quantize(
            "tiny",
            output_dir=tmp_path / "out",
            tuning=TUNING,
            calibration=corpus,
            memory_budget=1,
            memory_check=False,
        )
        predicted = json.loads(result.config_path.read_text(encoding="utf-8"))["memory_check"]
        assert predicted["fits"] is False
        assert predicted["enforced"] is False
        assert predicted["budget_bytes"] == 1
        assert predicted["budget_source"] == "caller"

    def test_an_unreadable_configuration_is_checked_after_the_load(
        self, corpus: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Offline with nothing cached, say: the loader's own configuration is
        # checked instead, before any weight is read.
        def unreadable(*_: Any) -> None:
            msg = "no network"
            raise ArchitectureError(msg)

        loads: list[str] = []

        def load(*_: Any, **__: Any) -> loader.ModelBundle:
            loads.append("loaded")
            return _bundle()

        monkeypatch.setattr(loader, "read_config", unreadable)
        monkeypatch.setattr(loader, "load_model", load)
        with pytest.raises(MemoryBudgetError):
            api.quantize(
                "tiny",
                output_dir=tmp_path / "out",
                tuning=TUNING,
                calibration=corpus,
                memory_budget=1,
            )
        assert loads == ["loaded"]

    def test_the_width_the_weights_loaded_at_corrects_the_configuration(
        self, corpus: str, tmp_path: Path
    ) -> None:
        # The configuration names no dtype, which reads as two bytes a weight;
        # the tiny model loads in float32, and the check is made again at four.
        result = api.quantize(
            "tiny", output_dir=tmp_path / "out", tuning=TUNING, calibration=corpus
        )
        predicted = json.loads(result.config_path.read_text(encoding="utf-8"))["memory_check"]
        assert predicted["model"]["dtype_bytes"] == 4

    def test_the_planning_pass_is_held_to_its_scoring_alone(self, corpus: str) -> None:
        info = api.plan_mixed_precision(
            "tiny", 3.0, (2, 4), scoring_samples=1, scoring_seq_len=16, calibration=corpus
        )
        assert set(info["memory_check"]["by_stage_bytes"]) == {"scoring"}

    def test_a_planning_pass_past_its_budget_is_refused_before_the_model_loads(
        self, corpus: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(loader, "load_model", _must_not_load)
        with pytest.raises(MemoryBudgetError, match="scoring"):
            api.plan_mixed_precision(
                "tiny",
                3.0,
                (2, 4),
                scoring_samples=1,
                scoring_seq_len=16,
                calibration=corpus,
                memory_budget=1,
            )
