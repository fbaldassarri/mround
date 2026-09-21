# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The run ledger: every measurement, machine-readable, appended forever.

MRound has a standing secondary mission of publishing its results (MEMORY.md
D-029), and a paper is built from numbers that can be cited, not from prose
recollections of them. This module is the difference between the two. Every
measurement harness appends one JSON line per run to a committed ledger, carrying
the full configuration, the outcomes, the costs, and the environment, so that any
figure quoted later is regenerable and attributable.

Three properties are load-bearing.

**Append-only, one line per run.** Nothing here rewrites history. A corrected
measurement is a new line whose ``notes`` say what it corrects, because a ledger
that can be edited in place is a ledger whose past cannot be trusted.

**Absent means unknown, never guessed.** Fields the harness could not observe are
``None``. Backfilled entries from before the ledger existed carry the log they
came from in ``source``, and their environment fields are only as complete as
the log was. A paper can live with "not recorded"; it cannot live with a
plausible value that was invented.

**Framework-free.** Importable and testable with no MLX and no model, like the
rest of the measurement arithmetic in this layer. Environment capture degrades
gracefully off-platform: the versions it cannot see are ``None``.

The ledger lives at ``benchmarks/ledger.jsonl`` and is committed, unlike the
checkpoint directories, because it is the project's data rather than its
outputs.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

__all__ = [
    "ENVIRONMENTS_DIR",
    "LEDGER_PATH",
    "RunRecord",
    "append_record",
    "capture_environment",
    "capture_packages",
    "read_ledger",
    "write_environment_manifest",
]

# Relative to the repository root. Committed, so it must not collide with the
# ignored output directories.
LEDGER_PATH = Path("benchmarks/ledger.jsonl")

# One file per distinct environment, named by the hash the ledger cites. Kept
# beside the ledger and committed with it: a measurement whose environment is
# not recoverable is not reproducible, whatever else was written down.
ENVIRONMENTS_DIR = Path("benchmarks/environments")

# A conda-meta filename is <name>-<version>-<build>.json, so a right split
# on two dashes yields exactly this many fields when it is one of ours.
_CONDA_META_FIELDS = 3

# Bumped when a field changes meaning. New fields may be added without a bump;
# readers must tolerate absence.
SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True, slots=True)
class RunRecord:
    """One measurement, with everything needed to cite or reproduce it.

    Attributes:
        kind: What was measured: ``rtn`` and ``quantize`` are written by
            ``examples/tune_model.py`` and ``round_to_nearest_model.py``,
            ``parity-reference`` by the reference comparison harness, and
            ``generate`` by hand from a terminal paste (three rows, whose
            evaluation and cost keys are their own). Readers filter on it
            rather than inferring.
        model: Hugging Face id or path of the model measured.
        scheme: Bits, group size, symmetry, as a plain mapping.
        tuning: The tuning configuration, or ``None`` for round-to-nearest.
        calibration: Corpus identity: source, packing, seed, samples, sequence
            length, and the content hash where known. The hash is what makes two
            runs provably comparable.
        evaluation: What was scored: ``dataset``, ``tokens`` (the count
            scored; live parity rows before 2026-09-17 hold the requested
            count, 65536, where 65504 were scored), ``seq_len``, ``stride``,
            and, from the same date, ``requested_tokens`` and ``dtype``.
        perplexity: Measured perplexities by label, such as ``original`` and
            ``quantized``. The backfilled parity row of 2026-08-13 keys
            MRound's side ``mround`` where the live rows key it ``quantized``.
        storage: Bits per weight and bytes on disk.
        cost: Wall-clock seconds by phase, peak memory bytes, and the device.
        environment: Machine and software versions at measurement time. Its
            ``packages`` field is the hash of the full pip and conda
            inventory, stored under :data:`ENVIRONMENTS_DIR`.
        source: Where the numbers came from: ``live`` for a harness writing at
            measurement time, or a log identifier for backfilled entries.
        notes: Anything the fields cannot say, such as what a correction
            corrects.
        recorded: UTC timestamp of the append, ISO 8601.
        schema: Schema version, for readers.
    """

    kind: str
    model: str
    scheme: dict[str, Any]
    tuning: dict[str, Any] | None = None
    calibration: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    perplexity: dict[str, float] | None = None
    storage: dict[str, Any] | None = None
    cost: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    source: str = "live"
    notes: str = ""
    recorded: str = ""
    schema: int = SCHEMA_VERSION


def capture_packages() -> dict[str, Any]:
    """Every installed package and its exact version, both channels.

    Two inventories, because a conda environment has two and neither is a
    superset of the other. ``pip`` covers importable distributions and is read
    from installed metadata rather than by shelling out. ``conda`` covers what
    conda placed in the prefix, including the non-Python libraries a numerical
    result can depend on, and is read from the ``conda-meta`` directory rather
    than through the ``conda`` executable: the filenames there already carry
    name, version and build, so this costs no subprocess and works when conda
    is not on the path.

    Returns:
        ``pip`` and ``conda`` name-to-version mappings, plus the environment's
        prefix and name where the interpreter exposes them. Either inventory
        is empty rather than absent when it does not apply, so a reader can
        tell "no conda" from "not recorded".
    """
    from importlib import metadata  # noqa: PLC0415

    pip: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pip[name.lower()] = dist.version or ""

    conda: dict[str, str] = {}
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        meta = Path(prefix) / "conda-meta"
        if meta.is_dir():
            for entry in sorted(meta.glob("*.json")):
                # <name>-<version>-<build>.json, and names may contain dashes,
                # so split from the right exactly twice.
                parts = entry.stem.rsplit("-", 2)
                if len(parts) == _CONDA_META_FIELDS:
                    conda[parts[0]] = f"{parts[1]}-{parts[2]}"

    return {
        "pip": dict(sorted(pip.items())),
        "conda": dict(sorted(conda.items())),
        "prefix": prefix,
        "name": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def write_environment_manifest(
    packages: dict[str, Any] | None = None,
    directory: str | Path = ENVIRONMENTS_DIR,
) -> str:
    """Record the full package inventory once, and return its content hash.

    The ledger carries one hash per run rather than several hundred package
    versions per run, and the inventory itself lives beside it under that
    hash. Two runs quoting the same hash are provably the same environment,
    down to the build string of every conda package; a run whose hash is new
    gets a new file and nothing is overwritten, because an environment that
    changed is a different environment rather than a correction to the old
    one.

    Args:
        packages: The inventory. ``None`` captures the current environment.
        directory: Where manifests live.

    Returns:
        The 16-character hash, for the ledger's ``environment.packages``.
    """
    payload = capture_packages() if packages is None else packages
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{digest}.json"
    if not target.exists():
        stamped = dict(payload)
        stamped["hash"] = digest
        stamped["first_seen"] = (
            datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
        )
        target.write_text(json.dumps(stamped, indent=2, sort_keys=True), encoding="utf-8")
    return digest


def capture_environment(
    *, packages: bool = True, directory: str | Path = ENVIRONMENTS_DIR
) -> dict[str, Any]:
    """What this measurement ran on, as far as it can be observed.

    Versions are read from installed package metadata rather than imported, so
    this works without loading MLX and returns ``None`` for anything absent
    rather than failing. The machine string is the hardware identity a paper's
    hardware table needs; on Apple Silicon ``platform.machine()`` alone says
    only ``arm64``, so the processor brand is included where the platform
    exposes it.

    The handful of named versions stay inline because they are the ones a
    reader wants without opening another file. Everything else is in the
    manifest the ``packages`` hash names, which is what makes a run
    reproducible rather than merely described.

    Args:
        packages: Write and reference the full package manifest. Off is for
            callers that must not touch the filesystem, and leaves the hash
            ``None`` rather than pretending the environment was captured.
        directory: Where the manifest goes. The default is relative to the
            working directory, which is the repository root for every harness;
            a caller that may run from elsewhere passes the absolute path.
    """
    from importlib import metadata  # noqa: PLC0415

    def version_of(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    return {
        "machine": platform.machine() or None,
        "processor": platform.processor() or None,
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "mlx": version_of("mlx"),
        "mlx_lm": version_of("mlx-lm"),
        "mround": version_of("mround"),
        "torch": version_of("torch"),
        "auto_round": version_of("auto-round"),
        "packages": write_environment_manifest(directory=directory) if packages else None,
    }


def append_record(record: RunRecord, path: str | Path = LEDGER_PATH) -> Path:
    """Append one run to the ledger, stamping the time if unstamped.

    Creates the ledger and its directory on first use. The write is a single
    line terminated by a newline, so a crash mid-run cannot corrupt earlier
    entries, only fail to add one.

    Args:
        record: The measurement.
        path: Ledger location. The default is the committed project ledger.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload = dataclasses.asdict(record)
    if not payload["recorded"]:
        payload["recorded"] = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat()
    line = json.dumps(payload, sort_keys=True, default=str)

    with destination.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return destination


def read_ledger(path: str | Path = LEDGER_PATH) -> list[dict[str, Any]]:
    """Every recorded run, oldest first.

    Returns plain dictionaries rather than :class:`RunRecord`, because old
    entries may predate fields the dataclass has since gained and readers must
    tolerate that rather than crash on history.

    Raises:
        ValueError: If a line is not valid JSON, naming the line number. A
            corrupt ledger is worth stopping over: silently skipping lines
            turns "every measurement" into "most measurements" without anyone
            deciding it.
    """
    source = Path(path)
    if not source.is_file():
        return []

    entries: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            msg = f"{source} line {number} is not valid JSON: {exc}"
            raise ValueError(msg) from exc
    return entries
