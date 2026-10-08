# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Retracting a ledger row appends and never edits, and refuses what it cannot mean."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from retract_row import build, main

from mround.eval.results import RunRecord, append_record, read_ledger


def _ledger(tmp_path: Path) -> Path:
    ledger = tmp_path / "ledger.jsonl"
    append_record(
        RunRecord(
            kind="parity-reference",
            model="m",
            scheme={"bits": 4},
            perplexity={"quantized": 1.0, "reference": 2.0},
        ),
        ledger,
    )
    return ledger


class TestBuild:
    def test_the_retraction_names_the_row_and_carries_no_measurement(self, tmp_path: Path) -> None:
        entries = read_ledger(_ledger(tmp_path))
        record = build(entries, 1, "two different models")
        assert record.kind == "retraction"
        assert record.source == "row 1"
        assert record.model == "m"
        assert record.scheme == {"bits": 4}
        assert record.perplexity is None
        assert record.notes == "Retraction of row 1. two different models"

    def test_a_missing_row_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            build(read_ledger(_ledger(tmp_path)), 2, "reason")

    def test_an_empty_reason_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="without a reason"):
            build(read_ledger(_ledger(tmp_path)), 1, "   ")

    def test_a_retraction_cannot_be_retracted(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        append_record(build(read_ledger(ledger), 1, "reason"), ledger)
        with pytest.raises(ValueError, match="carries no measurement"):
            build(read_ledger(ledger), 2, "reason")


class TestMain:
    def test_appends_once_and_refuses_the_second_time(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        before = ledger.read_text(encoding="utf-8")
        assert main(["1", "--reason", "two models", "--ledger", str(ledger)]) == 0
        after = ledger.read_text(encoding="utf-8")
        assert after.startswith(before), "a retraction appends and never edits"
        assert len(read_ledger(ledger)) == 2
        assert main(["1", "--reason", "again", "--ledger", str(ledger)]) == 1
        assert len(read_ledger(ledger)) == 2

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        ledger = _ledger(tmp_path)
        assert main(["1", "--reason", "two models", "--ledger", str(ledger), "--dry-run"]) == 0
        assert len(read_ledger(ledger)) == 1
