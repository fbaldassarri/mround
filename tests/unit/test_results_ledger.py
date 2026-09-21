# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The run ledger, which is the paper's raw data and must not lose or invent.

Framework-free like the module it tests. The properties that matter: appends
never disturb earlier lines, unknown stays None rather than becoming a guess,
old entries remain readable after the schema grows, and corruption stops the
reader instead of silently shrinking the dataset.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mround.eval.results import (
    RunRecord,
    append_record,
    capture_environment,
    capture_packages,
    read_ledger,
    write_environment_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


def minimal(kind: str = "rtn", notes: str = "") -> RunRecord:
    return RunRecord(
        kind=kind,
        model="test/model",
        scheme={"bits": 4, "group_size": 64, "symmetry": "sym"},
        notes=notes,
    )


class TestAppend:
    def test_round_trips(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        append_record(minimal(), ledger)
        (entry,) = read_ledger(ledger)
        assert entry["kind"] == "rtn"
        assert entry["scheme"]["bits"] == 4

    def test_appends_do_not_disturb_earlier_lines(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        append_record(minimal(notes="first"), ledger)
        before = ledger.read_text(encoding="utf-8")
        append_record(minimal(notes="second"), ledger)
        after = ledger.read_text(encoding="utf-8")
        # Append-only in the byte sense: history is a prefix of the new file.
        assert after.startswith(before)
        assert [e["notes"] for e in read_ledger(ledger)] == ["first", "second"]

    def test_the_timestamp_is_stamped_once(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        append_record(minimal(), ledger)
        stamped = read_ledger(ledger)[0]["recorded"]
        assert stamped  # filled in
        append_record(
            RunRecord(kind="rtn", model="m", scheme={}, recorded="2026-01-01T00:00:00"), ledger
        )
        # An explicit timestamp, as backfilled entries carry, is preserved.
        assert read_ledger(ledger)[1]["recorded"] == "2026-01-01T00:00:00"

    def test_unknown_fields_stay_none_not_guessed(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        append_record(minimal(), ledger)
        entry = read_ledger(ledger)[0]
        assert entry["perplexity"] is None
        assert entry["environment"] is None


class TestRead:
    def test_a_missing_ledger_is_empty_not_an_error(self, tmp_path: Path) -> None:
        assert read_ledger(tmp_path / "absent.jsonl") == []

    def test_old_entries_missing_new_fields_still_read(self, tmp_path: Path) -> None:
        # The reader must tolerate history written before a field existed,
        # which is why it returns dictionaries rather than the dataclass.
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text('{"schema": 1, "kind": "rtn", "model": "m"}\n', encoding="utf-8")
        assert read_ledger(ledger)[0]["model"] == "m"

    def test_corruption_stops_the_reader_and_names_the_line(self, tmp_path: Path) -> None:
        # Silently skipping a bad line turns "every measurement" into "most
        # measurements" without anyone deciding it.
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text('{"kind": "rtn"}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 2"):
            read_ledger(ledger)


class TestEnvironment:
    def test_capture_works_without_mlx_and_never_invents(self) -> None:
        captured = capture_environment(packages=False)
        assert captured["python"]
        # In an environment without these packages the honest answer is None.
        for key in ("mlx", "mlx_lm", "torch"):
            assert key in captured

    def test_capture_is_json_serializable(self) -> None:
        json.dumps(capture_environment(packages=False))

    def test_capture_can_decline_to_touch_the_filesystem(self) -> None:
        # A caller that must not write gets an honest None rather than a hash
        # for a manifest that was never written.
        assert capture_environment(packages=False)["packages"] is None


class TestEnvironmentManifest:
    """The full inventory, hashed once and cited many times.

    The ledger records what a run measured; the manifest records what it
    measured *with*. Keeping the inventory out of the row and behind a hash is
    what lets several hundred package versions be recoverable without making
    every row unreadable, and what lets two runs prove they shared an
    environment rather than merely claiming it.
    """

    def test_both_channels_are_reported_even_when_one_is_empty(self) -> None:
        packages = capture_packages()
        # Empty rather than absent: a reader must be able to tell "no conda
        # here" from "nobody looked".
        assert isinstance(packages["pip"], dict)
        assert isinstance(packages["conda"], dict)
        assert packages["pip"], "the interpreter running this has packages"

    def test_names_and_versions_are_plain_strings(self) -> None:
        packages = capture_packages()
        for name, version in packages["pip"].items():
            assert isinstance(name, str)
            assert isinstance(version, str)

    def test_the_hash_is_content_addressed_and_stable(self, tmp_path: Path) -> None:
        packages = capture_packages()
        first = write_environment_manifest(packages, tmp_path)
        second = write_environment_manifest(packages, tmp_path)
        assert first == second
        assert len(list(tmp_path.glob("*.json"))) == 1
        assert (tmp_path / f"{first}.json").is_file()

    def test_a_changed_environment_is_a_new_manifest_not_an_edit(self, tmp_path: Path) -> None:
        # An environment that changed is a different environment. Overwriting
        # would silently rewrite the provenance of every run that cited the
        # old hash.
        before = capture_packages()
        after = {**before, "pip": {**before["pip"], "numpy": "0.0.0-not-real"}}
        first = write_environment_manifest(before, tmp_path)
        second = write_environment_manifest(after, tmp_path)
        assert first != second
        assert len(list(tmp_path.glob("*.json"))) == 2
        recovered = json.loads((tmp_path / f"{first}.json").read_text())
        assert recovered["pip"] == before["pip"], "the first manifest was rewritten"

    def test_the_manifest_carries_its_own_hash_and_a_timestamp(self, tmp_path: Path) -> None:
        digest = write_environment_manifest(capture_packages(), tmp_path)
        stored = json.loads((tmp_path / f"{digest}.json").read_text())
        assert stored["hash"] == digest
        assert stored["first_seen"].endswith("+00:00")


class TestProjectLedger:
    def test_the_committed_ledger_parses_and_carries_provenance(self) -> None:
        # The backfilled history is paper data; a malformed line or an entry
        # with no source would make a number uncitable.
        from mround.eval.results import LEDGER_PATH  # noqa: PLC0415

        entries = read_ledger(LEDGER_PATH)
        if not entries:  # running from an installed copy without the data dir
            pytest.skip("project ledger not present here")
        for entry in entries:
            assert entry["schema"] == 1
            assert entry["source"], "an entry with no source is uncitable"
            assert entry["kind"]
            assert entry["model"]
