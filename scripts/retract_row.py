# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Mark a ledger row void, by appending, never by editing.

The ledger corrects by adding lines (MEMORY.md D-029). A wrong number gets a
correction row that carries the right one; a row that should never have been
written, because the thing it compares is not a comparison, gets a retraction:
a row of kind ``retraction`` whose ``source`` names the void row and whose
measurement fields are all ``None``. A reader that filters on kind drops the
named row and nothing else, and the void row stays where it is, so row numbers
cited elsewhere still point where they pointed.

The first use was row 80 (2026-09-24): a scoring run given ``--mround-dir`` for
a Qwen2.5-0.5B checkpoint while ``--model`` sat at the harness's SmolLM2
default, so the row's reference side and MRound side are two different models.

Usage, from the repository root::

    python scripts/retract_row.py 80 --reason "..." --dry-run
    python scripts/retract_row.py 80 --reason "..."

The reason is the retraction's whole note, so it should say what is wrong with
the row, which of its fields, if any, describe something real, and where the
right measurement lives or will live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mround.eval.results import LEDGER_PATH, RunRecord, append_record, read_ledger


def build(entries: list[dict[str, object]], row: int, reason: str) -> RunRecord:
    """The retraction record for ``row``, or a :class:`ValueError`."""
    if not 1 <= row <= len(entries):
        msg = f"row {row} does not exist; the ledger has {len(entries)} rows"
        raise ValueError(msg)
    target = entries[row - 1]
    if target.get("kind") in {"retraction", "provenance"}:
        msg = f"row {row} is a {target['kind']} row and carries no measurement to retract"
        raise ValueError(msg)
    already = [
        number
        for number, entry in enumerate(entries, start=1)
        if entry.get("kind") == "retraction" and entry.get("source") == f"row {row}"
    ]
    if already:
        msg = f"row {row} is already retracted by row {already[0]}"
        raise ValueError(msg)
    if not reason.strip():
        msg = "a retraction without a reason is an edit in disguise"
        raise ValueError(msg)
    return RunRecord(
        kind="retraction",
        model=str(target["model"]),
        scheme=dict(target.get("scheme") or {}),  # type: ignore[call-overload]
        source=f"row {row}",
        notes=f"Retraction of row {row}. {reason.strip()}",
    )


def main(argv: list[str] | None = None) -> int:
    """Append one retraction row, or print what would be appended."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("row", type=int, help="the row to mark void, counting from one")
    parser.add_argument(
        "--reason", required=True, help="what is wrong with it and where the truth is"
    )
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    entries = read_ledger(args.ledger)
    try:
        record = build(entries, args.row, args.reason)
    except ValueError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    print(f"row {len(entries) + 1}: retraction of row {args.row} ({record.model})")
    print(f"  {record.notes}")
    if not args.dry_run:
        append_record(record, args.ledger)
        print("appended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
