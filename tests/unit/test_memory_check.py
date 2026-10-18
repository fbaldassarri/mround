# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The memory check before a run: what it reads, what it asks, and what it says.

The rule itself is held to the recorded runs in ``test_memory_rule.py``. Here
is everything around it that a person meets: the configuration read without
the weights (``pipeline/loader.read_config``), the prediction without a run
(``api.predict_memory``), the command line's budget, its switch and its dry
run, and the device's memory as MLX reports it. None of it needs a model or a
Mac; the one test that needs MLX says so.
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from mround import api
from mround.cli.main import main
from mround.exceptions import ArchitectureError, MRoundError
from mround.pipeline import loader
from mround.planner.memory import ModelShape, RunShape, predict_memory
from mround.schemes import QuantScheme, ScaleInit, TuningConfig

if TYPE_CHECKING:
    from pathlib import Path

QWEN_7B: dict[str, Any] = {
    "model_type": "qwen2",
    "torch_dtype": "bfloat16",
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}
QWEN_3B: dict[str, Any] = {
    **QWEN_7B,
    "hidden_size": 2048,
    "intermediate_size": 11008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "tie_word_embeddings": True,
}


def _model_dir(tmp_path: Path, config: dict[str, Any], name: str = "model") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


@pytest.fixture
def no_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any test that loads a model: everything here reads a configuration."""

    def must_not_load(*_: Any, **__: Any) -> None:
        msg = "a model was loaded where only its configuration was needed"
        raise AssertionError(msg)

    monkeypatch.setattr(loader, "load_model", must_not_load)


def _fake_hub(monkeypatch: pytest.MonkeyPatch, download: Any) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    def hf_hub_download(*args: Any, **kwargs: Any) -> str:
        calls.append(args)
        return str(download(*args, **kwargs))

    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


class TestReadConfig:
    """A configuration read without the weights, from disk or the Hub."""

    def test_a_local_directory(self, tmp_path: Path) -> None:
        assert loader.read_config(_model_dir(tmp_path, QWEN_7B)) == QWEN_7B

    def test_a_local_file(self, tmp_path: Path) -> None:
        directory = _model_dir(tmp_path, QWEN_7B)
        assert loader.read_config(str(directory / "config.json")) == QWEN_7B

    def test_a_directory_without_one(self, tmp_path: Path) -> None:
        with pytest.raises(ArchitectureError, match="could not read"):
            loader.read_config(tmp_path)

    def test_a_file_that_is_not_json(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ArchitectureError, match="could not read"):
            loader.read_config(tmp_path)

    def test_json_that_is_not_a_configuration(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ArchitectureError, match="list"):
            loader.read_config(tmp_path)

    def test_a_repository_fetches_its_configuration_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        directory = _model_dir(tmp_path, QWEN_7B)
        calls = _fake_hub(monkeypatch, lambda *_, **__: directory / "config.json")
        assert loader.read_config("Qwen/Qwen2.5-7B-Instruct") == QWEN_7B
        assert calls == [("Qwen/Qwen2.5-7B-Instruct", "config.json")]

    def test_a_repository_that_cannot_be_read_says_why(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*_: Any, **__: Any) -> None:
            msg = "404 Client Error. Repository Not Found\nmore lines the Hub adds"
            raise OSError(msg)

        _fake_hub(monkeypatch, fail)
        with pytest.raises(ArchitectureError, match="Repository Not Found") as caught:
            loader.read_config("nobody/nothing")
        assert "more lines" not in str(caught.value)

    def test_without_the_hub_library_the_install_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(ArchitectureError, match="pip install"):
            loader.read_config("Qwen/Qwen2.5-7B-Instruct")


@pytest.mark.usefixtures("no_loading")
class TestPredictMemory:
    """The prediction of the run ``quantize`` would make, without making it."""

    def test_it_is_the_rule_on_the_run_quantize_would_resolve(self, tmp_path: Path) -> None:
        check = api.predict_memory(_model_dir(tmp_path, QWEN_7B), bits=4)
        shape = ModelShape.from_config(QWEN_7B)
        assert shape is not None
        # At 4 bits the scale starts from the observed range (D-046).
        run = RunShape.of(QuantScheme(bits=4), TuningConfig())
        assert not run.searched
        assert check.prediction == predict_memory(shape, run)

    def test_below_four_bits_the_search_is_predicted_too(self, tmp_path: Path) -> None:
        check = api.predict_memory(_model_dir(tmp_path, QWEN_7B), bits=2)
        assert check.prediction.run.searched
        scheme = QuantScheme(bits=2, scale_init=ScaleInit.SEARCHED)
        assert check.prediction.run == RunShape.of(scheme, TuningConfig())

    def test_a_mixed_run_is_predicted_with_its_scoring_pass(self, tmp_path: Path) -> None:
        check = api.predict_memory(_model_dir(tmp_path, QWEN_3B), average_bits=2.5)
        assert "scoring" in check.prediction.stages
        assert check.prediction.run.candidate_widths == 3

    def test_the_budget_and_the_pieces(self, tmp_path: Path) -> None:
        directory = _model_dir(tmp_path, QWEN_7B)
        check = api.predict_memory(directory, bits=2, memory_budget=29 * 10**9)
        assert check.budget == 29 * 10**9
        assert not check.fits
        assert check.micro_batches == 4
        pieces = api.predict_memory(
            directory,
            bits=2,
            tuning=TuningConfig(micro_batches=4),
            memory_budget=29 * 10**9,
        )
        assert pieces.fits

    def test_a_dense_remainder_rounds_nothing(self, tmp_path: Path) -> None:
        directory = _model_dir(tmp_path, QWEN_7B)
        rounded = api.predict_memory(directory, bits=2).prediction
        dense = api.predict_memory(directory, bits=2, dense_remainder=True).prediction
        assert dense.run.dense_remainder
        assert not rounded.run.dense_remainder
        assert dense.stages["remainder"] < rounded.stages["remainder"]
        assert dense.stages["export"] < rounded.stages["export"]
        for stage in ("entrance", "targets", "search", "score", "tune", "commit", "advance"):
            assert dense.stages[stage] == rounded.stages[stage], stage

    def test_a_model_the_rule_cannot_read_is_refused(self, tmp_path: Path) -> None:
        directory = _model_dir(tmp_path, {**QWEN_7B, "num_local_experts": 8})
        with pytest.raises(ArchitectureError, match="dense decoder"):
            api.predict_memory(directory)


@pytest.mark.usefixtures("no_loading")
class TestTheCommandLine:
    """The budget, the switch and the dry run."""

    def test_quantize_needs_an_output_unless_it_is_a_dry_run(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as caught:
            main(["quantize", "a-model"])
        assert caught.value.code == 2
        assert "--output-dir" in capsys.readouterr().err

    def test_a_dry_run_that_does_not_fit_names_the_pieces_and_fails(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = _model_dir(tmp_path, QWEN_7B)
        code = main(["quantize", str(directory), "-b", "2", "--dry-run", "--memory-budget", "29"])
        printed = capsys.readouterr()
        assert code == 1
        assert "predicted peak" in printed.out
        assert "tune" in printed.out
        assert "--micro-batches 4" in printed.err

    def test_a_dry_run_that_fits_succeeds(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        directory = _model_dir(tmp_path, QWEN_7B)
        argv = ["quantize", str(directory), "-b", "2", "--dry-run"]
        assert main([*argv, "--memory-budget", "29", "--micro-batches", "4"]) == 0
        assert "within the budget" in capsys.readouterr().out

    def test_a_dry_run_of_a_model_it_cannot_read_reports(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["quantize", str(tmp_path), "--dry-run"]) == 1
        assert capsys.readouterr().err.startswith("mround: ")

    @pytest.mark.parametrize("budget", ["0", "-3", "nan"])
    def test_a_budget_that_is_not_positive_is_refused(
        self, budget: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def must_not_run(*_: Any, **__: Any) -> None:
            msg = "quantize ran with a budget it should have refused"
            raise AssertionError(msg)

        monkeypatch.setattr(api, "quantize", must_not_run)
        assert main(["quantize", "a-model", "-o", "out", "--memory-budget", budget]) == 1
        assert "--memory-budget" in capsys.readouterr().err

    def test_the_budget_and_the_switch_reach_the_api(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        seen: dict[str, Any] = {}

        def quantize(*_: Any, **kwargs: Any) -> None:
            seen.update(kwargs)
            msg = "stopped after the arguments were seen"
            raise MRoundError(msg)

        monkeypatch.setattr(api, "quantize", quantize)
        argv = ["quantize", "a-model", "-o", "out"]
        assert main([*argv, "--memory-budget", "31.5", "--skip-memory-check"]) == 1
        assert seen["memory_budget"] == 31_500_000_000
        assert seen["memory_check"] is False
        capsys.readouterr()
        assert main(argv) == 1
        assert seen["memory_budget"] is None
        assert seen["memory_check"] is True


@pytest.mark.needs_mlx
class TestDeviceMemory:
    """The device's memory as MLX reports it, and zero where it does not."""

    def test_the_size_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mx = pytest.importorskip("mlx.core")
        from mround.pipeline import device  # noqa: PLC0415

        monkeypatch.setattr(mx, "device_info", lambda: {"memory_size": 34_359_738_368})
        assert device.device_memory() == 34_359_738_368

    @pytest.mark.parametrize("info", [{}, {"memory_size": 0}, {"memory_size": "lots"}])
    def test_nothing_reported_is_zero(
        self, info: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mx = pytest.importorskip("mlx.core")
        from mround.pipeline import device  # noqa: PLC0415

        monkeypatch.setattr(mx, "device_info", lambda: info)
        assert device.device_memory() == 0

    def test_an_accessor_that_fails_is_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mx = pytest.importorskip("mlx.core")
        from mround.pipeline import device  # noqa: PLC0415

        def fail() -> dict[str, Any]:
            msg = "no device"
            raise RuntimeError(msg)

        monkeypatch.setattr(mx, "device_info", fail)
        assert device.device_memory() == 0

    def test_this_build_answers_without_failing(self) -> None:
        pytest.importorskip("mlx.core")
        from mround.pipeline import device  # noqa: PLC0415

        assert device.device_memory() >= 0
