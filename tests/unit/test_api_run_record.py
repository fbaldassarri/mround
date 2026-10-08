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
from mround.exceptions import CalibrationError, SchemeError
from mround.pipeline import loader
from mround.pipeline.runner import QuantizationRunner
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
