# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Measure how much MRound's source resembles another implementation's.

CLAUDE.md forbids copying source from Intel AutoRound or anything else, and
says to stop if a function is being reproduced structurally line by line. That
is a rule about intent, and intent cannot be checked by a build. What can be
checked is the observable consequence: whether any function here has a near
twin over there. This script measures that and fails when it finds one.

Two views of every function, because either alone is easy to fool.

The **token view** is identifier blind. Every name that is not a keyword
becomes one placeholder, every number becomes another, every string a third,
and indentation is kept. Renaming variables is the cheapest way to disguise a
transcription and this view does not see renaming at all. What it does see is
the exact sequence of operators, calls, subscripts and keywords, which a
transcription preserves and an independent implementation does not.

The **shape view** is the abstract syntax tree walked in pre-order, node types
only. It ignores every name, constant and operator, and sees only the skeleton:
which statements nest inside which, in what order. It catches the case the
token view misses, where somebody kept the structure and changed the surface.

**Neither number means anything without a control.** Two independent
implementations of the same mathematics score high on both views simply because
the mathematics constrains the code: a grouped quantizer will reshape, compute a
maximum, divide, round and clamp whoever writes it. So this script compares
against as many trees as it is given, and the useful reading is not MRound's
score against the reference but the gap between that score and the same score
against code nobody could claim MRound was derived from. Run it with
``--control`` pointing at another library and read the distributions side by
side.

**A control has to be at least as large as the reference.** Every number here
is a maximum over a corpus, and a maximum grows with the number of things it is
taken over, so a small control makes any reference look damning. Measured:
against a 304 function library MRound's highest score is 0.741, against the
reference's 3459 it is 0.962, and against a 35308 function library it is 0.926,
all at the same floor. The first two differ because of size, not kinship. Pick
controls that are as big or bigger, and read the reported pool sizes before
reading anything else.

It never imports the tree it is comparing against. It parses the files, which
means it needs neither torch nor a GPU and runs anywhere the sources exist.

Usage:
    python scripts/measure_overlap.py --reference-path <dir> [--control <dir>]
"""

from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import difflib
import hashlib
import io
import json
import keyword
import statistics
import sys
import tokenize
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

__all__ = ["Match", "Unit", "collect", "compare", "main"]

# Functions below this many normalized tokens are not evidence of anything. A
# three line accessor, a one expression helper and a guard clause all look alike
# in every codebase ever written, and below this size the token view saturates:
# unrelated functions routinely score above 0.9 because there are not enough
# tokens for them to differ in. They are counted and reported, never gated.
#
# Sixty is measured rather than guessed. Sweeping the floor against three trees
# at once, the highest score MRound reaches anywhere falls from 0.962 at a floor
# of 24 to 0.880 at 40 and 0.749 at 60, and then stops moving: 80 gives 0.749
# again. That plateau is where the metric stops describing how short a function
# is and starts describing what is in it. MEMORY.md, 2026-09-20.
MIN_TOKENS = 60

# The default gate. At the calibrated floor nothing in MRound reaches 0.75
# against any tree, related or not, while an actual transcription scores above
# 0.95 on a view that cannot see renaming. Sitting the gate at 0.85 leaves a
# tenth of headroom for ordinary drift in either codebase and still fires long
# before a copied function could hide. Anything reaching it wants a human
# reading the pair side by side, not an automatic verdict.
THRESHOLD = 0.85

# How many of the closest pairs to print. Enough to see the shape of the top of
# the distribution, not so many that nobody reads it.
TOP = 15

# Token window for the candidate screen. Five is the usual shingle length for
# near duplicate detection and it holds up here despite the small alphabet
# these normalized tokens live in, because the screen only has to propose
# candidates: difflib decides.
SHINGLE = 5

# Windows appearing in more than this fraction of the corpus are dropped from
# the index. They are the idioms every Python file shares, so their posting
# lists are enormous and carry no signal.
COMMON = 0.02

# How many screened candidates get a real comparison. Beyond this the screen
# has already ranked the plausible twins above the noise, and every extra
# candidate is an edit distance nobody reads.
CANDIDATES = 40

_SKIP = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.ENCODING,
        tokenize.ENDMARKER,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Unit:
    """One function or method, reduced to the two forms the comparison uses.

    Attributes:
        origin: Label for the tree this came from, for the report.
        module: Path relative to that tree's root.
        name: Qualified name within the module, so a method is reported as
            ``ClassName.method`` rather than as a bare ``forward``.
        lineno: Where it starts, so a flagged pair can be opened immediately.
        tokens: Identifier blind token sequence.
        shape: Pre-order sequence of AST node type names.
    """

    origin: str
    module: str
    name: str
    lineno: int
    tokens: tuple[str, ...]
    shape: tuple[str, ...]

    @property
    def where(self) -> str:
        """One location string, as an editor would want it."""
        return f"{self.module}:{self.lineno} {self.name}"


@dataclasses.dataclass(frozen=True, slots=True)
class Match:
    """One MRound function and the closest thing to it in another tree.

    Attributes:
        unit: The MRound function.
        other: Its nearest neighbour, or ``None`` when the other tree held
            nothing comparable in size.
        token_ratio: Similarity on the identifier blind token view, 0 to 1.
        shape_ratio: Similarity on the AST shape view for that same pair. Not
            the best shape match available, deliberately: it describes the pair
            the token view chose, so the two numbers describe one comparison.
    """

    unit: Unit
    other: Unit | None
    token_ratio: float
    shape_ratio: float


def _normalized_tokens(source: str) -> list[tuple[int, str]]:
    """Tokenize a whole module once, normalized, tagged with line numbers.

    Whole module rather than per function because a function's source, sliced
    out and handed back to the tokenizer on its own, starts at a non zero
    indentation and raises. Tokenizing once and slicing by line is both correct
    and cheaper.

    Indentation is kept as its own token. Dropping it would flatten every
    function to a single block and inflate every score, because two pieces of
    code with the same operators in the same order but different nesting would
    become identical.
    """
    out: list[tuple[int, str]] = []
    try:
        stream = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in stream:
            if token.type in _SKIP:
                continue
            if token.type == tokenize.INDENT:
                out.append((token.start[0], "\x01"))
            elif token.type == tokenize.DEDENT:
                out.append((token.start[0], "\x02"))
            elif token.type == tokenize.NAME:
                out.append(
                    (token.start[0], token.string if keyword.iskeyword(token.string) else "N")
                )
            elif token.type == tokenize.NUMBER:
                out.append((token.start[0], "0"))
            elif token.type == tokenize.STRING:
                out.append((token.start[0], "S"))
            else:
                out.append((token.start[0], token.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # A file that does not tokenize is a file this comparison cannot speak
        # about. Returning nothing drops it rather than failing the run, which
        # matters because a large third party tree usually contains at least one
        # file written for a different Python version.
        return []
    return out


def _preorder(node: ast.AST) -> Iterator[str]:
    """AST node type names in pre-order.

    ``ast.walk`` is breadth first, which would put every sibling before every
    child and lose exactly the nesting this view exists to capture.
    """
    yield type(node).__name__
    for child in ast.iter_child_nodes(node):
        yield from _preorder(child)


def _functions(tree: ast.AST) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function and method, with a qualified name."""

    def walk(node: ast.AST, prefix: str) -> Iterator[tuple[str, Any]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                yield from walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                yield f"{prefix}{child.name}", child
                yield from walk(child, f"{prefix}{child.name}.")

    yield from walk(tree, "")


def collect(root: Path, origin: str, *, exclude: Sequence[str] = ()) -> list[Unit]:
    """Every function under a directory, in both comparison forms.

    Args:
        root: Directory to walk. Every ``.py`` file under it is read.
        origin: Label for the report.
        exclude: Path fragments that disqualify a file. Tests and generated
            code are excluded by the caller rather than here, because what
            counts as either differs per tree.

    Returns:
        One unit per function, skipping files that do not parse.
    """
    units: list[Unit] = []
    for path in sorted(root.rglob("*.py")):
        relative = str(path.relative_to(root))
        if any(fragment in relative for fragment in exclude):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (UnicodeDecodeError, SyntaxError, ValueError):
            continue
        tokens = _normalized_tokens(source)
        if not tokens:
            continue
        for name, node in _functions(tree):
            end = node.end_lineno or node.lineno
            sliced = tuple(text for line, text in tokens if node.lineno <= line <= end)
            if not sliced:
                continue
            units.append(
                Unit(
                    origin=origin,
                    module=relative,
                    name=name,
                    lineno=node.lineno,
                    tokens=sliced,
                    shape=tuple(_preorder(node)),
                )
            )
    return units


def _shingles(tokens: Sequence[str]) -> frozenset[int]:
    """Hashed overlapping windows of the token sequence.

    The screen that makes this affordable. Comparing every function here against
    every function there with a real edit distance is quadratic in both the
    number of functions and their length, and on trees this size that does not
    finish. Shingles reduce each function to a set, set intersection is cheap,
    and the expensive comparison then runs only on the handful of candidates
    that share enough windows to be worth the cost.

    **blake2b rather than the builtin hash, and that is not fastidiousness.**
    Python randomizes string hashing per process, so the builtin would give
    different integers on every run. Those integers decide which windows collide,
    which are common enough to drop from the index, what order a set iterates in,
    and therefore how ties break when the top candidates are chosen. The first
    version of this file used the builtin and two machines running identical
    source produced reports that differed in the tenth pair down. A gate that
    does not reproduce is not a gate: a borderline pair would pass on one machine
    and fail on another.
    """
    return frozenset(
        int.from_bytes(
            hashlib.blake2b(
                "\x00".join(tokens[index : index + SHINGLE]).encode("utf-8"), digest_size=8
            ).digest(),
            "big",
        )
        for index in range(len(tokens) - SHINGLE + 1)
    )


def _index(units: Sequence[Unit], sets: Sequence[frozenset[int]]) -> dict[int, list[int]]:
    """Map each shingle to the units containing it, dropping the common ones.

    A window like ``N . N ( N )`` appears in most Python ever written, and its
    posting list would be nearly the whole corpus, so it costs a great deal to
    look up and says nothing. Dropping the windows that appear in more than
    ``COMMON`` of the corpus keeps the index both small and informative.
    """
    postings: dict[int, list[int]] = {}
    for position, shingle_set in enumerate(sets):
        for shingle in shingle_set:
            postings.setdefault(shingle, []).append(position)
    ceiling = max(2, int(COMMON * len(units)))
    return {shingle: where for shingle, where in postings.items() if len(where) <= ceiling}


def compare(subjects: Sequence[Unit], others: Sequence[Unit], *, min_tokens: int) -> list[Match]:
    """Nearest neighbour in ``others`` for every subject above the size floor.

    Two stages. The shingle index proposes candidates by how many token windows
    they share with the subject, and difflib then scores the best few properly,
    because window overlap is a screen and not a similarity: it ignores order,
    and order is most of what distinguishes a transcription from a coincidence.

    Returns:
        One match per gated subject, sorted by token similarity, highest first.
    """
    pool = [unit for unit in others if len(unit.tokens) >= min_tokens]
    pool_sets = [_shingles(unit.tokens) for unit in pool]
    postings = _index(pool, pool_sets)

    matches: list[Match] = []
    for unit in subjects:
        if len(unit.tokens) < min_tokens:
            continue
        own = _shingles(unit.tokens)
        counts: collections.Counter[int] = collections.Counter()
        for shingle in own:
            where = postings.get(shingle)
            if where is not None:
                counts.update(where)
        if not counts:
            matches.append(Match(unit=unit, other=None, token_ratio=0.0, shape_ratio=0.0))
            continue

        # Ranked by containment rather than raw count, so that a short function
        # wholly contained in a long one outranks a long one sharing the same
        # number of windows by accident.
        #
        # The pool index is the tie break, and it is load bearing for the same
        # reason blake2b is: candidates tie constantly at this stage, only the
        # first CANDIDATES of them are scored properly, and without a total order
        # the cut falls wherever the dictionary happened to be built. Negated so
        # that ties resolve toward the earlier entry under ``reverse=True``.
        ranked = sorted(
            counts.items(),
            key=lambda item: (item[1] / max(1, min(len(pool_sets[item[0]]), len(own))), -item[0]),
            reverse=True,
        )[:CANDIDATES]

        matcher: difflib.SequenceMatcher[str] = difflib.SequenceMatcher(autojunk=False)
        matcher.set_seq2(unit.tokens)
        best = 0.0
        chosen = -1
        for position, _ in ranked:
            candidate = pool[position]
            bound = (
                2.0
                * min(len(candidate.tokens), len(unit.tokens))
                / (len(candidate.tokens) + len(unit.tokens))
            )
            if bound <= best:
                continue
            matcher.set_seq1(candidate.tokens)
            if matcher.real_quick_ratio() <= best or matcher.quick_ratio() <= best:
                continue
            ratio = matcher.ratio()
            if ratio > best:
                best = ratio
                chosen = position

        if chosen < 0:
            matches.append(Match(unit=unit, other=None, token_ratio=0.0, shape_ratio=0.0))
            continue
        other = pool[chosen]
        shape: difflib.SequenceMatcher[str] = difflib.SequenceMatcher(
            None, other.shape, unit.shape, autojunk=False
        )
        matches.append(Match(unit=unit, other=other, token_ratio=best, shape_ratio=shape.ratio()))
    matches.sort(key=lambda match: match.token_ratio, reverse=True)
    return matches


def _distribution(values: Iterable[float]) -> dict[str, float]:
    """Mean, median and the upper tail, which is the part that matters."""
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
        "p99": ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))],
        "max": ordered[-1],
    }


def _render(label: str, matches: Sequence[Match], *, pool: int, top: int, threshold: float) -> str:
    """The human readable block for one comparison.

    ``pool`` is printed first and deliberately: every score below it is a
    maximum taken over that many functions, and a reader comparing two blocks
    with very different pool sizes is not comparing like with like.
    """
    ratios = [match.token_ratio for match in matches]
    stats = _distribution(ratios)
    lines = [
        "",
        f"  {label}",
        f"  {'-' * len(label)}",
        f"    functions searched   {pool}",
        f"    functions gated      {len(matches)}",
    ]
    if stats:
        lines.append(
            f"    token similarity     mean {stats['mean']:.3f}  median {stats['median']:.3f}  "
            f"p90 {stats['p90']:.3f}  p99 {stats['p99']:.3f}  max {stats['max']:.3f}"
        )
    over = [match for match in matches if match.token_ratio >= threshold]
    lines.append(f"    at or above {threshold:.2f}     {len(over)}")
    lines.append("")
    lines.append("    closest pairs, mround first:")
    for match in matches[:top]:
        if match.other is None:
            continue
        flag = "  <<< OVER" if match.token_ratio >= threshold else ""
        lines.append(
            f"      {match.token_ratio:.3f} tokens  {match.shape_ratio:.3f} shape   "
            f"{match.unit.where}{flag}"
        )
        lines.append(f"{'':>30}against  {match.other.where}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="measure_overlap",
        description="Measure MRound's source similarity against another implementation.",
    )
    parser.add_argument(
        "--reference-path",
        required=True,
        help="Directory holding the reference implementation's sources, normally the "
        "installed auto_round package. It is parsed, never imported.",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="DIR",
        help="Another tree to compare against, for the null distribution. Repeatable. "
        "Without at least one, the reference number has nothing to be read against.",
    )
    parser.add_argument(
        "--source",
        default="mround",
        help="The tree being checked (default: mround).",
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    parser.add_argument("--top", type=int, default=TOP)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison.

    Returns:
        Zero when nothing reached the threshold against the reference, one when
        something did, two when the inputs were unusable.
    """
    args = build_parser().parse_args(argv)

    source_root = Path(args.source)
    reference_root = Path(args.reference_path)
    for root in (source_root, reference_root):
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2

    # Tests and examples are excluded from the subject side: a test that pins a
    # formula necessarily restates it, and a comparison that flags its own test
    # suite teaches nobody anything.
    subjects = collect(source_root, args.source, exclude=("tests/", "examples/"))
    if not subjects:
        print(f"no functions found under {source_root}", file=sys.stderr)
        return 2

    targets: list[tuple[str, list[Unit]]] = [
        (f"{reference_root.name} (the reference)", collect(reference_root, reference_root.name))
    ]
    for control in args.control:
        root = Path(control)
        if not root.is_dir():
            print(f"not a directory: {root}", file=sys.stderr)
            return 2
        targets.append((f"{root.name} (control)", collect(root, root.name)))

    results: list[tuple[str, int, list[Match]]] = []
    for label, units in targets:
        pool = sum(1 for unit in units if len(unit.tokens) >= args.min_tokens)
        results.append((label, pool, compare(subjects, units, min_tokens=args.min_tokens)))

    reference_matches = results[0][2]
    over = [match for match in reference_matches if match.token_ratio >= args.threshold]

    if args.json:
        payload: dict[str, Any] = {
            "source": str(source_root),
            "threshold": args.threshold,
            "min_tokens": args.min_tokens,
            "functions": len(subjects),
            "comparisons": {
                label: {
                    "searched": pool,
                    "gated": len(matches),
                    "distribution": _distribution(m.token_ratio for m in matches),
                    "over_threshold": [
                        {
                            "mround": m.unit.where,
                            "other": m.other.where if m.other else None,
                            "tokens": m.token_ratio,
                            "shape": m.shape_ratio,
                        }
                        for m in matches
                        if m.token_ratio >= args.threshold
                    ],
                }
                for label, pool, matches in results
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1 if over else 0

    small = sum(1 for unit in subjects if len(unit.tokens) < args.min_tokens)
    print("")
    print("mround source overlap")
    print("")
    print(f"  functions in {args.source}   {len(subjects)}")
    print(f"  below {args.min_tokens} tokens, not gated   {small}")
    for label, pool, matches in results:
        print(_render(label, matches, pool=pool, top=args.top, threshold=args.threshold))
    print("")
    if over:
        print(f"  FAIL  {len(over)} at or above {args.threshold:.2f} against the reference.")
        print("        Read each pair side by side before doing anything else. A high score is")
        print("        evidence to be explained, not a verdict: the mathematics forces some")
        print("        agreement, which is what the control column is there to quantify.")
    else:
        print(f"  ok    nothing at or above {args.threshold:.2f} against the reference.")
    print("")
    return 1 if over else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
