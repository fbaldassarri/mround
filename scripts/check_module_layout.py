# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify the package tree matches DOCUMENTATION.md section 4.

DOCUMENTATION.md is the authoritative technical reference. When it and the code
disagree, one of them is wrong and the disagreement must be resolved rather than
tolerated, so this checks rather than trusts.

Parses the module layout diagram out of DOCUMENTATION.md and compares it against
what is on disk, in both directions: a documented module that does not exist is
a broken promise, and an undocumented module is a layer someone added without
saying so.

Run standalone or from CI::

    python scripts/check_module_layout.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "DOCUMENTATION.md"
PACKAGE_ROOT = REPO_ROOT / "mround"

# Files present for packaging or tooling reasons rather than as architecture.
IGNORED = {"__init__.py", "py.typed"}


def documented_modules() -> set[str]:
    """Extract module paths from the layout diagram in section 4.

    The diagram is a fenced block drawn with box characters. Every line naming a
    ``.py`` file contributes one path, reconstructed from the indentation depth
    of the directory lines above it.

    Returns:
        Paths relative to the package root, such as ``core/quantizer.py``.

    Raises:
        SystemExit: If the diagram cannot be located, which means the document
            changed shape and this script needs updating with it.
    """
    text = DOC_PATH.read_text()
    match = re.search(r"```\n(mround/\n.*?)```", text, re.DOTALL)
    if match is None:
        sys.stderr.write("could not find the module layout diagram in DOCUMENTATION.md\n")
        raise SystemExit(2)

    # The fenced block also draws the tests/ and scripts/ trees. Only the
    # package subtree is ours to check, and it ends at the first blank line.
    package_tree = match.group(1).split("\n\n", maxsplit=1)[0]

    modules: set[str] = set()
    stack: list[tuple[int, str]] = []

    for raw in package_tree.splitlines()[1:]:
        # Strip the box-drawing prefix, keeping its width so depth survives.
        cleaned = re.sub(r"[│├└─]", " ", raw)
        name_match = re.match(r"^(\s*)([\w.]+(?:\.py)?)/?", cleaned)
        if name_match is None:
            continue
        indent, name = len(name_match.group(1)), name_match.group(2)

        while stack and stack[-1][0] >= indent:
            stack.pop()

        if name.endswith(".py"):
            prefix = "/".join(part for _, part in stack)
            modules.add(f"{prefix}/{name}" if prefix else name)
        else:
            stack.append((indent, name))

    return {m for m in modules if Path(m).name not in IGNORED}


def actual_modules() -> set[str]:
    """Every Python module on disk, relative to the package root."""
    return {
        str(path.relative_to(PACKAGE_ROOT))
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.name not in IGNORED
    }


def main() -> int:
    """Compare the documented layout against the tree.

    Returns:
        Zero when they agree, one when they do not.
    """
    documented = documented_modules()
    actual = actual_modules()

    missing = sorted(documented - actual)
    undocumented = sorted(actual - documented)

    for path in missing:
        print(f"::error::documented in DOCUMENTATION.md but absent from the tree: {path}")
    for path in undocumented:
        print(f"::error::present in the tree but undocumented in DOCUMENTATION.md: {path}")

    if missing or undocumented:
        print(f"\n{len(missing)} missing, {len(undocumented)} undocumented")
        return 1

    print(f"module layout matches DOCUMENTATION.md ({len(actual)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
