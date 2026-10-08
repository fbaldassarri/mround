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
import re
from pathlib import Path
from typing import Any

import pytest

from mround.eval.results import (
    ENVIRONMENTS_DIR,
    SCHEMA_VERSION,
    RunRecord,
    append_record,
    capture_environment,
    capture_packages,
    peak_resident_bytes,
    read_ledger,
    write_environment_manifest,
)

# The ledger and its manifests are found from this file rather than from the
# working directory, which LEDGER_PATH and ENVIRONMENTS_DIR are relative to.
REPO_ROOT = Path(__file__).resolve().parents[2]


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

    def test_an_absent_package_is_none_not_a_guess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The test above checks only that the keys exist, so a version
        # invented for an absent package would pass it.
        from importlib import metadata  # noqa: PLC0415

        def absent(name: str) -> str:
            raise metadata.PackageNotFoundError(name)

        monkeypatch.setattr(metadata, "version", absent)
        captured = capture_environment(packages=False)
        for key in ("mlx", "mlx_lm", "mround", "torch", "auto_round"):
            assert captured[key] is None, key

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


class TestSchemaTwo:
    def test_new_fields_default_to_unrecorded(self, tmp_path: Path) -> None:
        # A row written by a harness that observed neither block says so with
        # None rather than an empty mapping a reader could mistake for "nothing
        # to report".
        ledger = tmp_path / "ledger.jsonl"
        append_record(minimal(), ledger)
        (entry,) = read_ledger(ledger)
        assert entry["schema"] == SCHEMA_VERSION == 2
        assert entry["reference_environment"] is None
        assert entry["memory"] is None
        assert entry["tasks"] is None, "only a zeroshot row carries task results"

    def test_a_schema_one_row_reads_without_the_new_fields(self, tmp_path: Path) -> None:
        ledger = tmp_path / "ledger.jsonl"
        ledger.write_text(
            '{"schema": 1, "kind": "parity-reference", "model": "m", "source": "live"}\n',
            encoding="utf-8",
        )
        (entry,) = read_ledger(ledger)
        assert "reference_environment" not in entry, "the reader must not invent a field"

    def test_peak_resident_bytes_is_bytes_or_none(self) -> None:
        peak = peak_resident_bytes()
        if peak is None:
            pytest.skip("this platform does not expose the resident set")
        # A process running a test suite has resident memory in the megabytes.
        # On Linux ru_maxrss is in kilobytes and on macOS in bytes; a value
        # below a megabyte here would mean the unit was not normalized.
        assert peak >= 1 << 20


class TestProjectLedger:
    # The head to head rows written before schema 2, whose reference arm's
    # environment lived only in their notes. Each one has exactly one
    # provenance row restating it (rows 55 onward, appended 2026-09-23).
    SCHEMA_ONE_HEAD_TO_HEAD = frozenset({9, *range(29, 52), 54})

    @staticmethod
    def _entries() -> list[dict[str, Any]]:
        from mround.eval.results import LEDGER_PATH  # noqa: PLC0415

        entries = read_ledger(REPO_ROOT / LEDGER_PATH)
        if not entries:  # a copy of the tree without the data dir
            pytest.skip("project ledger not present here")
        return entries

    def test_the_committed_ledger_parses_and_carries_provenance(self) -> None:
        # The backfilled history is paper data; a malformed line or an entry
        # with no source would make a number uncitable.
        for entry in self._entries():
            assert entry["schema"] in (1, 2)
            assert entry["source"], "an entry with no source is uncitable"
            assert entry["kind"]
            assert entry["model"]

    def test_retractions_name_a_real_measurement_row_and_carry_none_themselves(self) -> None:
        # A retraction is the ledger's only way to say a row is void. It must
        # point at a row that exists and measured something, carry no
        # measurement of its own, and say why.
        entries = self._entries()
        for number, row in enumerate(entries, start=1):
            if row["kind"] != "retraction":
                continue
            target = int(row["source"].removeprefix("row "))
            assert 1 <= target < number, "a retraction names an earlier row"
            assert entries[target - 1]["kind"] not in ("provenance", "retraction")
            assert row["model"] == entries[target - 1]["model"]
            for field in ("perplexity", "cost", "evaluation", "calibration", "storage", "memory"):
                assert row[field] is None
            assert row["notes"].startswith(f"Retraction of row {target}.")

    def test_no_head_to_head_row_the_paper_cites_is_retracted(self) -> None:
        # The published results cite rows by number in two ranges: 1 to 79,
        # the provenance rows 55 to 79 among them, and 82 to 107, the zero
        # shot rows 97 to 104 among those. A retraction of any row in either
        # would have to reach RESULTS.md and the paper too. Row 80, retracted
        # by row 81, lies between them and must be cited by neither.
        entries = self._entries()
        retracted = {
            int(row["source"].removeprefix("row "))
            for row in entries
            if row["kind"] == "retraction"
        }
        cited = set(range(1, 80)) | set(range(82, 108))
        assert not (retracted & cited)

    def test_history_is_intact(self) -> None:
        # Provenance rows were appended, never inserted: the row numbers the
        # paper cites (`paper/figures.py` indexes by them) must still name the
        # measurements they named, and every schema 1 row is still schema 1.
        entries = self._entries()
        assert len(entries) >= 54
        for number, entry in enumerate(entries[:54], start=1):
            assert entry["schema"] == 1, f"row {number} was rewritten"
            assert entry["kind"] != "provenance"
        head_to_head = {
            number
            for number, entry in enumerate(entries[:54], start=1)
            if entry["kind"] == "parity-reference"
        }
        assert head_to_head == self.SCHEMA_ONE_HEAD_TO_HEAD

    def test_every_schema_one_head_to_head_row_has_one_provenance_row(self) -> None:
        entries = self._entries()
        provenance = [e for e in entries if e["kind"] == "provenance"]
        cited = [int(e["source"].removeprefix("row ")) for e in provenance]
        assert sorted(cited) == sorted(self.SCHEMA_ONE_HEAD_TO_HEAD), (
            "each schema 1 head to head row is restated exactly once"
        )
        for row in provenance:
            assert row["schema"] == 2
            for field in ("perplexity", "cost", "evaluation", "calibration", "storage", "memory"):
                assert row[field] is None, f"a provenance row carries no {field}"
            assert row["environment"] is None, "the machine that restated a fact is not part of it"
            block = row["reference_environment"]
            assert set(block) == {
                "packages",
                "auto_round",
                "torch",
                "python",
                "system",
                "machine",
                "processor",
                "host",
                "device",
            }
            assert block["device"] == "cpu", "every reference arm before 2026-09-23 ran on a CPU"
            assert block["auto_round"] in ("0.14.2", "0.15.0")
            assert row["notes"], "a provenance row says where each field came from"

    def test_provenance_rows_agree_with_the_rows_they_restate(self) -> None:
        # The restated block is only as good as its agreement with what the
        # original row already said in its cost field and its notes.
        entries = self._entries()
        for row in (e for e in entries if e["kind"] == "provenance"):
            original = entries[int(row["source"].removeprefix("row ")) - 1]
            block = row["reference_environment"]
            assert row["model"] == original["model"]
            assert row["scheme"] == original["scheme"]
            cost = original.get("cost") or {}
            if cost.get("reference_host"):
                assert block["host"] == cost["reference_host"]
            if block["packages"] is not None:
                # Row 44 is a correction line that repeats row 30's fields and
                # cites only its own Linux timing arm; the hash of the Mac
                # checkpoint it scored is in row 30's notes.
                cited = original["notes"] + (
                    entries[29]["notes"] if row["source"] == "row 44" else ""
                )
                assert block["packages"] in cited, (
                    "the environment hash must be the one the original row cited"
                )
                manifest = REPO_ROOT / ENVIRONMENTS_DIR / f"{block['packages']}.json"
                assert manifest.is_file(), "a cited manifest is committed beside the ledger"
                inventory = json.loads(manifest.read_text(encoding="utf-8"))
                assert inventory["pip"]["auto-round"] == block["auto_round"]
                assert inventory["pip"]["torch"] == block["torch"]
            if block["auto_round"] == "0.15.0":
                # The whole reason schema 2 exists: these rows read 0.14.2 in
                # their own environment block, which describes the scorer.
                assert original["environment"]["auto_round"] == "0.14.2"

    # Rows 55 to 116 in the order they were appended. The paper and RESULTS.md
    # cite most of them by number, so their kinds are history as much as the
    # head to head rows of SCHEMA_ONE_HEAD_TO_HEAD are.
    SCHEMA_TWO_KINDS = (
        ("provenance",) * 25  # rows 55 to 79
        + ("parity-reference", "retraction")  # rows 80 and 81
        + ("parity-reference",) * 15  # rows 82 to 96
        + ("zeroshot",) * 8  # rows 97 to 104
        + ("parity-reference",) * 6  # rows 105 to 110
        + ("zeroshot",) * 6  # rows 111 to 116
    )
    KINDS = frozenset(
        {"rtn", "quantize", "generate", "parity-reference", "provenance", "retraction", "zeroshot"}
    )

    def test_rows_55_to_116_are_schema_2_and_keep_their_kinds(self) -> None:
        entries = self._entries()
        assert len(entries) >= 116
        assert {entry["kind"] for entry in entries} <= self.KINDS, "a reader filters on kind"
        assert tuple(entry["kind"] for entry in entries[54:116]) == self.SCHEMA_TWO_KINDS
        for number, entry in enumerate(entries[54:], start=55):
            assert entry["schema"] >= 2, f"row {number} predates schema 2"
            if entry["kind"] == "parity-reference":
                # Written under schema 2, so the reference arm's environment
                # is the row's own block rather than a provenance row's.
                assert entry["reference_environment"], f"row {number}"

    # The two Qwen2.5-0.5B-Instruct tables again, with the reference's own
    # SignRound v2 tuning on, each row beside the default arm of the same
    # seed: the 2 bit table (rows 105 to 107) and the mixed 2.5 one (rows 108
    # to 110).
    OWN_TUNING_TABLES = (
        ((105, 88), (106, 33), (107, 89)),
        ((108, 43), (109, 45), (110, 46)),
    )
    OWN_TUNING_TWINS = tuple(pair for table in OWN_TUNING_TABLES for pair in table)
    # A row of those tables whose twin was written under schema 1, beside a
    # schema 2 row scored against the same MRound checkpoint: the Arc arms of
    # seed 43, at 2 bits and at mixed 2.5.
    SAME_MROUND_CHECKPOINT = ((106, 90), (109, 84))
    # Where a head to head row keeps what belongs to its reference arm, and
    # the three fields that belong to the row itself. Everything else a
    # default twin records, the row with the reference's own tuning repeats.
    REFERENCE_ARM = (
        "cost.reference_",
        "memory.reference_",
        "perplexity.reference",
        "reference_environment",
    )
    OF_THE_ROW = frozenset({"recorded", "notes", "schema"})

    @staticmethod
    def _shared_scheme(row: dict[str, Any]) -> dict[str, Any]:
        """A row's scheme without MRound's own allocation, which follows the seed."""
        return {key: value for key, value in row["scheme"].items() if key != "layers_by_width"}

    @classmethod
    def _leaves(cls, value: Any, prefix: str = "") -> dict[str, Any]:
        """A row as dotted paths to its values, so two rows compare field by field."""
        if not isinstance(value, dict):
            return {prefix[:-1]: value}
        leaves: dict[str, Any] = {}
        for key, inner in value.items():
            leaves.update(cls._leaves(inner, f"{prefix}{key}."))
        return leaves

    def test_a_row_names_the_reference_s_own_tuning_in_its_block_and_its_notes(self) -> None:
        # The reference keeps its SignRound v2 tuning behind enable_alg_ext,
        # and rows 105 to 110 are the first whose reference arm turned it on.
        # The block says so as the class the release resolved and the notes
        # say so in words; a row carrying one without the other would be read
        # as a default arm by whoever filtered on the other.
        entries = self._entries()
        flagged = []
        for number, row in enumerate(entries, start=1):
            if row["kind"] not in ("parity-reference", "zeroshot"):
                continue
            block = row.get("reference_environment") or {}
            # The class name is the reference's own, spelled here because the
            # harness that reads it back is not part of every tree this test
            # runs in; a release that renames the class changes this line too.
            in_block = block.get("algorithm") == "SignRoundV2Config"
            in_notes = "enable_alg_ext" in (row.get("notes") or "")
            assert in_block == in_notes, f"row {number}"
            if in_block:
                flagged.append(number)
        own_tuning = [number for number, _ in self.OWN_TUNING_TWINS]
        assert flagged[: len(own_tuning)] == own_tuning, "no earlier arm ran it"
        for number in own_tuning:
            block = entries[number - 1]["reference_environment"]
            # Under 0.15.0 that tuning runs its block forward uncompiled while
            # the default arms compiled it, so a time beside a twin's names it.
            assert block["auto_round"] == "0.15.0"
            assert block["block_forward_compile"] is False
            assert block["torch_compile"] is True

    def test_the_own_tuning_rows_differ_from_their_twins_in_the_reference_arm_alone(
        self,
    ) -> None:
        # Each was scored against the MRound checkpoint its default twin was
        # scored against, which is what lets the two be read side by side. So
        # whatever the twin records outside its reference arm, the row repeats
        # to the last digit: the model, seed, recipe, scheme, evaluation and
        # scoring machine, the original and MRound figures, MRound's size and
        # time, and in the mixed table the two allocations, MRound's and the
        # reference's, counted by width. What differs is the reference arm,
        # and in two things, both of which the documents name: its tuning, and
        # the system it was produced on, since the Linux machine changed
        # operating system release between the default arms and these with
        # one package environment.
        entries = self._entries()
        assert len(entries) >= 110
        provenance = {
            int(row["source"].removeprefix("row ")): row["reference_environment"]
            for row in entries
            if row["kind"] == "provenance"
        }
        first_of_its_table = {
            number: entries[table[0][0] - 1]
            for table in self.OWN_TUNING_TABLES
            for number, _ in table
        }
        first_of_all = entries[self.OWN_TUNING_TWINS[0][0] - 1]
        for number, twin_number in self.OWN_TUNING_TWINS:
            row, twin = entries[number - 1], entries[twin_number - 1]
            ours = self._leaves(row)
            # Rows 33 and 43 recorded the token count asked for under tokens.
            # From row 45 on a head to head row records the count scored
            # there and the request beside it, whatever its schema.
            asked_only = "requested_tokens" not in twin["evaluation"]
            for key, value in self._leaves(twin).items():
                if key in self.OF_THE_ROW or key.startswith(self.REFERENCE_ARM):
                    continue
                mine = (
                    "evaluation.requested_tokens"
                    if key == "evaluation.tokens" and asked_only
                    else key
                )
                assert mine in ours, f"row {number} records no {mine}"
                assert ours[mine] == value, f"row {number} {mine}"
            # A schema 1 twin records less than its row does. What it cannot
            # vouch for is held to an earlier row of the six. The scheme is
            # held to the first row of its table, MRound's own allocation
            # apart, which follows the seed and which every mixed twin
            # records. The recipe, the evaluation and the setup of the
            # reference arm are held to the first row of all: one setup
            # produced the six arms and one evaluation scored them.
            first = first_of_its_table[number]
            assert self._shared_scheme(row) == self._shared_scheme(first), f"row {number} scheme"
            for field in ("tuning", "evaluation", "reference_environment"):
                assert row[field] == first_of_all[field], f"row {number} {field}"
            # The six MRound checkpoints were produced before a run recorded
            # its memory, so the two fields a schema 2 row keeps for MRound's
            # are null in every one of these rows.
            for key in ("mround_peak_bytes", "mround_rss_bytes"):
                assert row["memory"][key] is None, f"row {number} {key}"

            assert row["perplexity"]["reference"] != twin["perplexity"]["reference"]
            assert row.get("tasks") is None, "a head to head row carries no task results"
            # A schema 1 twin (rows 33, 43, 45 and 46) keeps its reference
            # arm's environment in the provenance row that restates it.
            block = row["reference_environment"]
            twin_block = twin.get("reference_environment") or provenance[twin_number]
            for key in ("packages", "auto_round", "torch", "python", "machine", "device"):
                assert block[key] == twin_block[key], f"row {number} reference {key}"
            # The precision and the compiler are in a twin's block from
            # 2026-09-24 on, so the provenance row of a schema 1 twin has
            # neither.
            for key in ("precision", "torch_compile"):
                if key in twin_block:
                    assert block[key] == twin_block[key], f"row {number} reference {key}"
            assert "algorithm" not in twin_block, "the twin ran the default tuning"
            # The two systems, as the documents name them. A row records its
            # reference arm's host twice, in its block and beside its cost.
            assert block["host"] == "Linux x86_64 Ubuntu 26.04.1 LTS", f"row {number}"
            assert block["system"] == "Linux 7.0.0-38-generic", f"row {number}"
            assert row["cost"]["reference_host"] == block["host"], f"row {number}"
            assert block["host"] in row["notes"], f"row {number}"
            assert "Ubuntu 24.04" in twin_block["host"], f"row {twin_number}"
            assert twin_block["system"] == "Linux 7.0.0-31-generic", f"row {twin_number}"
            assert twin["cost"]["reference_host"] == twin_block["host"], f"row {twin_number}"
        # A schema 1 twin has no field for MRound's memory or for whether its
        # recipe was per layer, and row 33 has none for MRound's scoring time
        # or for its time in detail. Rows 90 and 84, the Arc arms of seed 43,
        # were scored against the MRound checkpoints of rows 106 and 109 under
        # schema 2 and have every one of those fields, so the two rows are
        # held to them: the same figures, and null where the sibling is null.
        for number, sibling in self.SAME_MROUND_CHECKPOINT:
            ours, theirs = self._leaves(entries[number - 1]), self._leaves(entries[sibling - 1])
            assert ours["perplexity.quantized"] == theirs["perplexity.quantized"]
            for key, value in theirs.items():
                if key.startswith(("cost.mround_", "memory.mround_", "scheme.mround_")):
                    assert ours[key] == value, f"row {number} {key}"

    # Zero shot rows of the six pairs with the reference's own tuning, each
    # beside the zero shot row of its default twin. The two head to head rows
    # of a pair were scored against one MRound checkpoint, so the suite scored
    # that checkpoint twice, eight days apart.
    SUITE_SCORED_TWICE = ((111, 98), (112, 99), (113, 100), (114, 101), (115, 102), (116, 103))

    def test_a_checkpoint_scored_twice_on_the_suite_repeats_its_figures(self) -> None:
        # The documents read the task suite as deterministic on Metal: the
        # original scored identically beside every comparison, and MRound's
        # six checkpoints, scored again for rows 111 to 116, repeated rows 98
        # to 103 to the last digit, so the new rows' only new figures are the
        # reference's. That holds while these pairs agree field for field.
        entries = self._entries()
        assert len(entries) >= 116
        for number, earlier in self.SUITE_SCORED_TWICE:
            row, before = entries[number - 1], entries[earlier - 1]
            assert row["kind"] == before["kind"] == "zeroshot", f"row {number}"
            cited = row["evaluation"]["head_to_head_row"]
            twin = before["evaluation"]["head_to_head_row"]
            assert (cited, twin) in self.OWN_TUNING_TWINS, f"row {number} cites row {cited}"
            mround = entries[cited - 1]["perplexity"]["quantized"]
            assert mround == entries[twin - 1]["perplexity"]["quantized"], f"row {number}"
            assert row["evaluation"]["suite"] == before["evaluation"]["suite"]
            for arm in ("original", "quantized"):
                assert row["tasks"][arm] == before["tasks"][arm], f"row {number} {arm}"
            assert row["tasks"]["reference"] != before["tasks"]["reference"], f"row {number}"

    def test_every_zero_shot_row_repeats_the_head_to_head_it_cites(self) -> None:
        # A zero shot row scores the three checkpoints of one head to head row
        # and repeats its configuration, which is what lets a reader join the
        # two (RunRecord, mround/eval/results.py). Rows 1 to 96 predate the
        # tasks field and have no such key at all, hence get.
        entries = self._entries()
        retracted = {
            int(row["source"].removeprefix("row "))
            for row in entries
            if row["kind"] == "retraction"
        }
        zero_shot = [
            (number, row)
            for number, row in enumerate(entries, start=1)
            if row["kind"] == "zeroshot"
        ]
        assert len(zero_shot) >= 8, "rows 97 to 104"
        for number, row in zero_shot:
            cited = row["evaluation"]["head_to_head_row"]
            assert 1 <= cited < number, f"row {number} cites a later row"
            original = entries[cited - 1]
            assert original["kind"] == "parity-reference", f"row {number} cites row {cited}"
            assert cited not in retracted, f"row {number} cites a retracted row"
            assert re.search(rf"\brow {cited}\b", row["notes"]), f"row {number}"
            for field in ("model", "scheme", "calibration", "tuning"):
                assert row[field] == original[field], f"row {number} {field} is not row {cited}'s"
            assert row["perplexity"] is None
            assert original.get("tasks") is None
            tasks = row.get("tasks") or {}
            assert set(tasks) == {"original", "reference", "quantized"}, f"row {number}"
            for arm in tasks.values():
                assert set(arm) == set(row["evaluation"]["suite"]), f"row {number}"
