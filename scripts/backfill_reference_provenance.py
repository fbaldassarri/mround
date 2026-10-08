# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Restate the reference environment of every schema 1 head to head row.

Ledger schema 1 recorded, on each ``parity-reference`` row, the environment of
the machine that scored the row, and named the machine that produced the
reference arm only in free text: a hash in ``notes`` and a host string in
``cost.reference_host``. A reader parsing the published ledger got the scoring
machine's ``auto_round: "0.14.2"`` on every reference arm that 0.15.0 produced.
Schema 2 gives the reference arm its own block (``reference_environment``,
``mround/eval/results.py``), and this script fills that block in for the rows
written before it existed, the way this project corrects anything: by
appending one ``provenance`` row per head to head row, never by editing one
(MEMORY.md D-029).

Every field comes from the sidecar the producing machine wrote beside its
checkpoint, ``parity_reference.json``, which is checked against the row it
restates before anything is written: same model, same seed, the host string
the row's cost field recorded, the environment hash the row's notes cited, and
the reference wall clock where both recorded it. A disagreement stops the
script, because a provenance row that guessed would be worse than the gap it
fills. Row 9 predates the sidecar and is restated from its own fields; row 44
is a correction line whose perplexity came from the same Mac checkpoint as row
30, and its Linux timing arm is named in the note.

``device`` is the one field no sidecar of that period carries. It is ``cpu``
on every row, for reasons the note of each row states: on the Linux machine
the sidecar's own torch build is ``2.13.0+cpu``, which has no accelerator to
run on (MEMORY.md D-039); on the Mac the reference's device manager redirects
to the CPU unless ``--device_map=mps`` is passed, which the harness never did
(MEMORY.md, 2026-09-07, and ``build_reference`` in the harness).

Usage, from the repository root on the machine that holds the sidecars::

    python scripts/backfill_reference_provenance.py --dry-run
    python scripts/backfill_reference_provenance.py

``--sidecars`` points at the directory the ``parity-*`` output directories live
under when it is not the repository root. The script refuses to run twice: a
ledger that already carries a provenance row for any mapped row is left alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mround.eval.results import (
    ENVIRONMENTS_DIR,
    LEDGER_PATH,
    RunRecord,
    append_record,
    read_ledger,
)

# Row number, counting from one as the paper cites them, to the sidecar of the
# reference arm that row scored, relative to the repository root. Several rows
# scored one checkpoint more than once (rows 32, 38 and 39; rows 40 and 47;
# rows 29, 30 and 44), so several rows share a sidecar. ``None`` marks the one
# row with no sidecar, whose harness predated it.
SIDECARS: dict[int, str | None] = {
    9: None,
    29: "parity-reference/parity_reference.json",
    30: "parity-reference/parity_reference.json",
    31: "parity-qwen2.5-0.5b-instruct-2bit-g64/reference/parity_reference.json",
    32: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3"
        "/reference/parity_reference.json"
    ),
    33: "parity-qwen2.5-0.5b-instruct-2bit-g64-s43-ar0.15.0/reference/parity_reference.json",
    34: "parity-qwen2.5-0.5b-instruct-2bit-g64-s43/reference/parity_reference.json",
    35: "parity-smollm2-135m-instruct-2bit-g64-denserem/reference/parity_reference.json",
    36: "parity-qwen2.5-1.5b-instruct-2bit-g64/reference/parity_reference.json",
    37: "parity-qwen2.5-0.5b-instruct-2bit-g64-s44/reference/parity_reference.json",
    38: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3"
        "/reference/parity_reference.json"
    ),
    39: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3"
        "/reference/parity_reference.json"
    ),
    40: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024"
        "/reference/parity_reference.json"
    ),
    41: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-s43"
        "/reference/parity_reference.json"
    ),
    42: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-s44"
        "/reference/parity_reference.json"
    ),
    43: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-cost"
        "/reference/parity_reference.json"
    ),
    44: "parity-reference/parity_reference.json",
    45: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-s43-ar015"
        "/reference/parity_reference.json"
    ),
    46: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-s44-ar015"
        "/reference/parity_reference.json"
    ),
    47: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024"
        "/reference/parity_reference.json"
    ),
    48: (
        "parity-qwen2.5-0.5b-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-best"
        "/reference/parity_reference.json"
    ),
    49: (
        "parity-smollm2-135m-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024"
        "/reference/parity_reference.json"
    ),
    50: (
        "parity-smollm2-135m-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-s43"
        "/reference/parity_reference.json"
    ),
    51: (
        "parity-smollm2-135m-instruct-avg2.5bit-o2-3-4-g64-widest-r3-sc128x1024-s44"
        "/reference/parity_reference.json"
    ),
    54: "parity-smollm2-135m-instruct-4bit-g64/reference/parity_reference.json",
}

# The row 44 correction line names a second reference production, on Linux,
# whose timer it carries as ``reference_seconds_linux``. Its checkpoint never
# reached the Mac, so there is no sidecar; its environment hash is in the row's
# notes and its versions are in the committed manifest under that hash.
LINUX_ARM_OF_ROW_44 = "9e6019673ac3629b"
CORRECTION_ROW = 44
CORRECTED_ROW = 30

CPU_ON_LINUX = (
    "device is cpu because the sidecar's torch build is {torch}, a build with no "
    "accelerator to run on (MEMORY.md D-039); the sidecar predates the device field."
)
# The published ledger must not name the private harness by file, so the
# note says what it did rather than where it lives.
CPU_ON_MAC = (
    "device is cpu because the reference's device manager redirects to the CPU on "
    "Apple Silicon unless --device_map=mps is passed, which the comparison harness "
    "never did (MEMORY.md, 2026-09-07, 'The first head to head'); the sidecar "
    "predates the device field."
)


class ProvenanceError(ValueError):
    """A sidecar and the row it should restate disagree, so nothing is written."""


def _manifest_versions(digest: str) -> dict[str, str]:
    """The reference release and torch build a committed manifest records."""
    manifest = json.loads((ENVIRONMENTS_DIR / f"{digest}.json").read_text(encoding="utf-8"))
    conda_python = manifest["conda"].get("python", "")
    return {
        "auto_round": manifest["pip"]["auto-round"],
        "torch": manifest["pip"]["torch"],
        "python": conda_python.split("-")[0] if conda_python else "",
    }


def _check(condition: bool, row: int, what: str) -> None:
    if not condition:
        msg = f"row {row}: {what}"
        raise ProvenanceError(msg)


def _from_sidecar(
    row: int,
    original: dict[str, Any],
    sidecars: Path,
    relative: str,
    corrected: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """The provenance block for ``row``, checked against the row it restates.

    ``corrected`` is the row a correction line repeats its fields from: row 44
    cites only its Linux timing arm in its own notes, and the hash of the Mac
    checkpoint it scored is in row 30's.
    """
    sidecar = json.loads((sidecars / relative).read_text(encoding="utf-8"))
    environment = sidecar["environment"]
    config = sidecar["config"]
    cost = original.get("cost") or {}
    notes = original["notes"]
    cited = notes if corrected is None else notes + corrected["notes"]

    _check(config["model"] == original["model"], row, "the sidecar names another model")
    _check(config["seed"] == original["calibration"]["seed"], row, "the sidecar has another seed")
    _check(sidecar["host"] == cost["reference_host"], row, "the sidecar names another host")
    _check(environment["packages"] in cited, row, "the row's notes cite another environment")
    seconds = sidecar.get("seconds")
    if seconds is not None and cost.get("reference_seconds") is not None:
        _check(seconds == cost["reference_seconds"], row, "the wall clocks disagree")
    manifest = _manifest_versions(environment["packages"])
    _check(manifest["auto_round"] == environment["auto_round"], row, "manifest vs sidecar release")
    _check(manifest["torch"] == environment["torch"], row, "manifest vs sidecar torch")

    on_linux = environment["system"].startswith("Linux")
    if on_linux:
        _check(environment["torch"].endswith("+cpu"), row, "a Linux torch build that is not +cpu")
        device_reason = CPU_ON_LINUX.format(torch=environment["torch"])
    else:
        _check(environment["system"].startswith("Darwin"), row, "an unrecognized system")
        device_reason = CPU_ON_MAC

    block = {
        "packages": environment["packages"],
        "auto_round": environment["auto_round"],
        "torch": environment["torch"],
        "python": environment["python"],
        "system": environment["system"],
        "machine": environment["machine"],
        "processor": environment["processor"],
        "host": sidecar["host"],
        "device": "cpu",
    }
    agreement = (
        "host, environment hash, model and seed agree with the row"
        if seconds is None
        else "host, environment hash, wall clock, model and seed agree with the row"
    )
    if corrected is not None:
        agreement += (
            f" (the hash with row {CORRECTED_ROW}'s notes, the row this correction line "
            "repeats its fields from)"
        )
    note = (
        f"Provenance row for row {row}, appended under ledger schema 2 on 2026-09-23. "
        f"It restates where row {row}'s reference arm was produced, which schema 1 "
        "recorded only in the row's notes and cost.reference_host while its environment "
        "block described the scoring machine. Every field but device is read from the "
        f"sidecar the producing machine wrote beside the checkpoint ({relative}), whose "
        f"{agreement}. {device_reason} The reference release is auto-round "
        f"{environment['auto_round']}. No measurement is carried here; the numbers stay in "
        f"row {row}."
    )
    if row == CORRECTION_ROW:
        linux = _manifest_versions(LINUX_ARM_OF_ROW_44)
        _check(LINUX_ARM_OF_ROW_44 in notes, row, "the Linux arm's hash is not in the notes")
        note += (
            " Row 44 is the cost correction to row 30 and scored the same Mac checkpoint, "
            "so this block describes that Mac production; its reference_seconds_linux came "
            f"from a second production on {cost['reference_host_linux']} under environment "
            f"{LINUX_ARM_OF_ROW_44} (auto-round {linux['auto_round']}, torch {linux['torch']}, "
            f"python {linux['python']}, log00001-ubuntu), on the CPU for the same D-039 "
            "reason, whose checkpoint never reached the scoring machine and has no sidecar here."
        )
    return block, note


def _row_nine(row: int, original: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Row 9 is backfilled from log0028 and predates the sidecar entirely."""
    environment = original["environment"]
    cost = original["cost"]
    _check(cost.get("reference_device") == "cpu", row, "the row does not say cpu")
    _check(environment["machine"] == "arm64", row, "the row is not the Mac")
    block = {
        "packages": None,
        "auto_round": environment["auto_round"],
        "torch": environment["torch"],
        "python": environment["python"],
        "system": environment["system"],
        "machine": environment["machine"],
        "processor": environment["processor"],
        "host": "Darwin arm64",
        "device": "cpu",
    }
    note = (
        "Provenance row for row 9, appended under ledger schema 2 on 2026-09-23. Row 9 is "
        "backfilled from log0028 and no sidecar exists for it: the harness did not write one "
        "in August. Both arms ran on the same Mac, so the block restates the row's own "
        "environment fields (system, machine, processor, python, torch, auto_round) as the "
        "log recorded them, python to the minor version only because that is all the log "
        "gave. packages is None because hashed environment manifests (D-038) began on "
        "2026-09-06. host is that machine in the token the later rows use for it. device is "
        "the row's own cost.reference_device, and MEMORY.md 2026-08-11 records 3102.6 seconds "
        "on the CPU with the reference's own device resolution redirecting away from MPS. "
        "No measurement is carried here; the numbers stay in row 9."
    )
    return block, note


def build_rows(entries: list[dict[str, Any]], sidecars: Path) -> list[RunRecord]:
    """One provenance record per mapped row, or a :class:`ProvenanceError`."""
    records: list[RunRecord] = []
    for row, relative in sorted(SIDECARS.items()):
        original = entries[row - 1]
        _check(original["kind"] == "parity-reference", row, "not a head to head row")
        _check(original["schema"] == 1, row, "not a schema 1 row")
        if relative is None:
            block, note = _row_nine(row, original)
        else:
            corrected = entries[CORRECTED_ROW - 1] if row == CORRECTION_ROW else None
            block, note = _from_sidecar(row, original, sidecars, relative, corrected)
        records.append(
            RunRecord(
                kind="provenance",
                model=original["model"],
                scheme=original["scheme"],
                reference_environment=block,
                source=f"row {row}",
                notes=note,
            )
        )
    return records


def main(argv: list[str] | None = None) -> int:
    """Check every sidecar against its row, then append, or only print."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument(
        "--sidecars",
        type=Path,
        default=Path(),
        help="directory the parity-* output directories live under (default: here)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the rows, write nothing")
    args = parser.parse_args(argv)

    entries = read_ledger(args.ledger)
    if len(entries) < max(SIDECARS):
        print(
            f"{args.ledger} has {len(entries)} rows; row {max(SIDECARS)} is mapped", file=sys.stderr
        )
        return 1
    already = {
        int(entry["source"].removeprefix("row "))
        for entry in entries
        if entry.get("kind") == "provenance"
    }
    if already & set(SIDECARS):
        print(f"provenance rows already present for rows {sorted(already & set(SIDECARS))}")
        return 1

    try:
        records = build_rows(entries, args.sidecars)
    except ProvenanceError as exc:
        print(f"refusing to write: {exc}", file=sys.stderr)
        return 1

    for record in records:
        block = record.reference_environment or {}
        print(
            f"{record.source:>7}: {block['host']}, auto-round {block['auto_round']}, "
            f"torch {block['torch']}, {block['device']}, environment {block['packages']}"
        )
        if not args.dry_run:
            append_record(record, args.ledger)
    print(f"{'would append' if args.dry_run else 'appended'} {len(records)} provenance rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
