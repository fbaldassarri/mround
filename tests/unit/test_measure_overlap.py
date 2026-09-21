# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The similarity check's own behaviour, which decides whether a release ships.

This is the one tool in the repository whose output is an accusation, so both
of its failure modes are expensive. A check that cannot see a transcription
lets one through; a check that fires on ordinary code trains everybody to
ignore it. These tests pin both ends: a copy with every name changed still
scores 1.0, and two functions doing different work do not.

They deliberately do not pin the threshold or the floor. Those are calibrated
against real corpora and recorded in MEMORY.md, and a test that froze them here
would make recalibration look like a regression.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from measure_overlap import Unit, collect, compare

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

ORIGINAL = '''
def quantize_group(weight, bits, group_size):
    """Docstring here."""
    groups = weight.reshape(-1, group_size)
    limit = 2 ** (bits - 1) - 1
    scale = groups.abs().max(axis=1) / limit
    codes = (groups / scale).round().clip(-limit - 1, limit)
    return codes, scale
'''

# The same function with every identifier, every constant and the docstring
# changed. A transcription disguised exactly as far as renaming can disguise it.
RENAMED = '''
def pack_block(tensor, width, block):
    """Something else entirely."""
    chunks = tensor.reshape(-1, block)
    bound = 2 ** (width - 1) - 1
    factor = chunks.abs().max(axis=1) / bound
    indices = (chunks / factor).round().clip(-bound - 1, bound)
    return indices, factor
'''

# Same domain, same vocabulary, genuinely different work: no reshaping, no
# scale, a loop instead of an expression.
DIFFERENT = '''
def summarize_widths(layers, target):
    """Count how the plan spent its budget."""
    counts = {}
    total = 0
    for name, width in layers.items():
        counts[width] = counts.get(width, 0) + 1
        total += width
    if not layers:
        raise ValueError("no layers")
    return counts, total / len(layers) - target
'''


def _write(tmp_path: Path, name: str, body: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "module.py").write_text(body, encoding="utf-8")
    return root


def _only(units: list[Unit]) -> Unit:
    assert len(units) == 1, [unit.name for unit in units]
    return units[0]


def test_renaming_everything_does_not_hide_a_copy(tmp_path: Path) -> None:
    """The point of the identifier blind view, stated as a test."""
    original = collect(_write(tmp_path, "a", ORIGINAL), "a")
    renamed = collect(_write(tmp_path, "b", RENAMED), "b")

    assert _only(original).tokens == _only(renamed).tokens
    matches = compare(original, renamed, min_tokens=1)
    assert matches[0].token_ratio == pytest.approx(1.0)
    assert matches[0].shape_ratio == pytest.approx(1.0)


def test_different_work_scores_well_below_a_copy(tmp_path: Path) -> None:
    """A check that flagged this pair would be worthless."""
    original = collect(_write(tmp_path, "a", ORIGINAL), "a")
    different = collect(_write(tmp_path, "b", DIFFERENT), "b")

    matches = compare(original, different, min_tokens=1)
    assert matches[0].token_ratio < 0.7


def test_the_size_floor_excludes_small_functions(tmp_path: Path) -> None:
    """Small functions are counted, never gated: they saturate the metric."""
    original = collect(_write(tmp_path, "a", ORIGINAL), "a")
    tokens = len(_only(original).tokens)

    assert compare(original, original, min_tokens=tokens) != []
    assert compare(original, original, min_tokens=tokens + 1) == []


def test_an_empty_corpus_yields_no_neighbour(tmp_path: Path) -> None:
    """A tree with nothing comparable reports no match rather than a zero."""
    original = collect(_write(tmp_path, "a", ORIGINAL), "a")
    empty = collect(_write(tmp_path, "b", "X = 1\n"), "b")

    match = compare(original, empty, min_tokens=1)[0]
    assert match.other is None
    assert match.token_ratio == 0.0


def test_methods_are_reported_with_their_class(tmp_path: Path) -> None:
    """A bare ``forward`` in a report names nothing a reader can open."""
    source = "class Outer:\n    def inner(self):\n        return 1\n"
    units = collect(_write(tmp_path, "a", source), "a")
    assert _only(units).name == "Outer.inner"


def test_files_that_do_not_parse_are_skipped(tmp_path: Path) -> None:
    """A large third party tree usually contains at least one such file."""
    root = tmp_path / "a"
    root.mkdir()
    (root / "broken.py").write_text("def (:\n", encoding="utf-8")
    (root / "fine.py").write_text(ORIGINAL, encoding="utf-8")
    assert [unit.module for unit in collect(root, "a")] == ["fine.py"]


def _shingles_in_a_fresh_process(seed: str) -> str:
    program = (
        "import sys;sys.path.insert(0, sys.argv[1]);"
        "from measure_overlap import _shingles;"
        "print(sorted(_shingles(tuple('a b c d e f g h'.split()))))"
    )
    environment = dict(os.environ, PYTHONHASHSEED=seed)
    result = subprocess.run(
        [sys.executable, "-c", program, str(SCRIPTS)],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    return result.stdout


def test_the_screen_is_stable_across_processes() -> None:
    """The defect this test exists for shipped once and must not ship twice.

    The first version hashed token windows with the builtin ``hash``, which
    Python randomizes per process. Those integers decide which windows collide,
    which are dropped as too common, and how ties break when candidates are
    chosen, so two machines running identical source produced reports that
    differed part way down the list. A gate whose verdict depends on the
    interpreter's hash seed is not a gate.

    Two fresh processes with deliberately different seeds, because the failure
    cannot be reproduced inside one process, where the seed is fixed at startup.
    """
    assert _shingles_in_a_fresh_process("0") == _shingles_in_a_fresh_process("12345")


def test_nesting_changes_the_score(tmp_path: Path) -> None:
    """Indentation is a token, so the same statements nested differently differ.

    Dropping indentation would flatten every function to one block and make two
    pieces of code with the same operators in a different structure identical,
    which is the exact case the shape view is meant to catch.
    """
    flat = "def f(a, b):\n    x = a + b\n    y = x * 2\n    return y\n"
    nested = "def f(a, b):\n    x = a + b\n    if x:\n        y = x * 2\n        return y\n"
    one = collect(_write(tmp_path, "a", flat), "a")
    two = collect(_write(tmp_path, "b", nested), "b")
    assert compare(one, two, min_tokens=1)[0].token_ratio < 1.0
