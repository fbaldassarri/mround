# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Record the current environment's full package inventory as a manifest.

Run inside the environment whose runs are being recorded::

    conda activate mround-dev
    python scripts/capture_environment.py

The measurement harnesses do this automatically for every scored run, so this
script exists for the two cases they do not cover: capturing the environment
before or between runs, and checking whether the environment has drifted since
the last one, which the hash answers in one line.

Prints the hash the ledger cites, where the manifest went, and how many
packages each channel contributed. A hash that matches the previous run's is
the same environment; a hash that differs is not, and the difference is
recoverable by diffing the two files.
"""

from __future__ import annotations

from mround.eval.results import (
    ENVIRONMENTS_DIR,
    capture_packages,
    read_ledger,
    write_environment_manifest,
)


def main() -> int:
    """Capture, store, and report against whatever the ledger last cited."""
    packages = capture_packages()
    digest = write_environment_manifest(packages)

    print(f"environment {digest}")
    print(f"  manifest   {ENVIRONMENTS_DIR / f'{digest}.json'}")
    print(f"  conda env  {packages['name'] or 'none'} at {packages['prefix'] or 'no prefix'}")
    print(f"  packages   {len(packages['pip'])} pip, {len(packages['conda'])} conda")

    previous = [
        entry["environment"]["packages"]
        for entry in read_ledger()
        if isinstance(entry.get("environment"), dict) and entry["environment"].get("packages")
    ]
    if not previous:
        print("  no earlier run cites a manifest, so this is the first")
    elif previous[-1] == digest:
        print("  unchanged since the last recorded run")
    else:
        print(f"  CHANGED since the last recorded run, which cited {previous[-1]}")
        print("  Runs either side of this point are not environment-comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
