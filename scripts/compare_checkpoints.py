# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Where two fake quantized checkpoints of one model differ, code by code.

Two productions of one reference arm that differ in a single thing, the device
they tuned on (row 45's CPU arm against row 84's XPU arm, MEMORY.md
2026-09-24), export the same shapes at the same per layer widths. The
perplexity says how far apart the two landed; this says how. The reference's
fake export holds the dequantized floats, and for a symmetric integer scheme
every value in a group is an integer multiple of that group's scale, so the
scale is recoverable from the values themselves: the smallest positive
magnitude or gap in the group. With the scale in hand, two groups either
quantized on the same lattice, in which case their codes can be compared one
by one and the count of codes that moved is a number with a meaning, or they
did not, in which case the learned clipping drifted apart and every value
differs slightly, which is a different finding.

The last figure printed is the one the comparison harness prints between
MRound and the reference, the mean over layers of the root mean square
difference relative to the left side's own root mean square, so the two are on
one scale. What it found the first time
it ran (MEMORY.md 2026-09-24): two reference arms that differ only in the
device they tuned on differ in 64 percent of their weights, on different
lattices in 94 percent of their groups, and by 0.22 in that figure at mixed
2.5, which is what a change of release (0.22) or of seed (0.24) also does,
while the same seed on the same device reproduces the checkpoint to the bit,
across a change of operating system as well (MEMORY.md 2026-10-09), across
the change of release from auto-round 0.15.0 to 0.16.0 (2026-10-10), and on
the integrated GPU across a change of its compute runtime (2026-10-15); the
change of release that moved it was the one from 0.14.2 to 0.15.0.
A change of the reference's tuning algorithm is a different event (MEMORY.md
2026-10-02): at 2 bits its default checkpoint and its SignRound v2 checkpoint
of one seed share a lattice in 0.3 percent of their groups, with the scales
elsewhere a median of 14 to 15 percent of the default checkpoint's apart (17
of the other's), and sit 0.40 to 0.41 apart with the default checkpoint on the
left (0.43 with it on the right, both figures being relative to the left
side), where two seeds of either tuning sit 0.30 to 0.34 apart. At mixed 2.5
(MEMORY.md 2026-10-03) the same pair shares a lattice in 0.7 percent of its
groups, with the scales elsewhere a median of 9 to 10 percent apart, and sits
0.284 to 0.285 apart (0.296 to 0.297 with the sides swapped), where two seeds
of either tuning sit 0.22 to 0.24 apart. Layer by layer the figure follows the
width, 0.40 at the 2 bit layers and 0.22 at the 3 bit ones there, so a mixed
pair's figure is a mean over its widths.

MRound's own checkpoints are read too (since 2026-10-11, for M1b's A/B,
MEMORY.md). They hold no dequantized floats: each quantized layer is MLX's
packed form, ``uint32`` words of codes with a ``scales`` and a ``biases``
tensor beside them, and ``config.json``'s ``quantization`` block gives each
layer's width and group size. Nothing has to be inferred there. Two such
layers share a lattice in a group when their stored scales and biases agree,
and their stored codes are compared one by one; the relative disagreement is
taken over the dequantized values, ``scales * codes + biases``, which are
exact in float64. Against a fake export, a packed side is dequantized and its
stored scales stand in for the ones the other side's values imply. A layer a
packed checkpoint left dense holds its source's weights rather than a
quantization, so it is counted apart from the table.

Usage, from the repository root::

    python scripts/compare_checkpoints.py LEFT RIGHT --group-size 64

``LEFT`` and ``RIGHT`` are checkpoint directories holding ``config.json`` and
``model.safetensors``, or a parent with one such directory under it, which is
how the reference writes its output. ``--group-size`` is the fake exports'
group size, 64 when it is not given; a packed checkpoint carries its own, and
a group size given against it has to agree with it. Needs numpy and nothing
else: the safetensors header is read directly and one tensor at a time, so the
two checkpoints never have to fit in memory together.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Two scales this close are one scale written by two float32 computations; a
# code comparison across them is exact. Beyond it the groups are on different
# lattices and their codes are not compared, only counted.
SAME_LATTICE = 1e-3
# Reported separately: the groups whose scales agree to the last digit, which
# is what a bit identical trajectory would leave behind.
IDENTICAL_SCALE = 1e-6
# A value this far from the nearest integer multiple of the scale is not on
# the lattice at all, which for a fake export means the scale was misread.
INTEGRALITY = 1e-2
# Codes are integers, so any two that differ do so by at least one step.
HALF_STEP = 0.5
MATRIX = 2
WORST_SHOWN = 5
# No integer scheme is wider than this, so the search for a width stops here
# whatever the codes are.
WIDEST = 32
# The group size of a fake export when none is given, the reference's default.
DEFAULT_GROUP_SIZE = 64
# MLX packs codes into words of this many bits, least significant field first.
WORD_BITS = 32
# How a packed layer's words are stored in the safetensors header.
PACKED = "U32"
# The reference's distance from itself, measured with this script on the
# ledger's own checkpoints and published in RESULTS.md, printed beside every
# result so that a figure is read against the floor for its width rather than
# taken as a verdict on its own.
SELF_DISAGREEMENT = (
    "\n  Read against the reference's disagreement with itself (RESULTS.md): 0.30 to\n"
    "  0.34 at 2 bits across a change of release (0.14.2 to 0.15.0) or seed, 0.22\n"
    "  to 0.24 at mixed 2.5 across a change of device, release or seed, 0.095 at 4\n"
    "  bits across a change of device or precision, and 0.0000 for one seed\n"
    "  produced again on one device, even after a change of its operating system,\n"
    "  of its GPU runtime or of release from 0.15.0 to 0.16.0.\n"
    "  A change of its tuning algorithm at one seed is 0.40 to 0.41 at 2 bits with\n"
    "  the default arm on the left and 0.43 with it on the right, and 0.284 to\n"
    "  0.285 and 0.296 to 0.297 at mixed 2.5: the figure is relative to the left\n"
    "  side."
)

_DTYPES = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
# Exports the reference writes after tuning under autocast hold bfloat16
# weights: eight significant bits, so a dequantized value is rounded by up to
# one part in 256, and a gap between two adjacent codes at the top of a 4 bit
# range by fifteen times that. The lattice can no longer be read off the
# values, so for a side stored this way only the float figures are reported.
_COARSE = {"BF16", "F16"}
_ARCHITECTURE_FIELDS = ("model_type", "hidden_size", "num_hidden_layers", "vocab_size")


@dataclass
class TensorEntry:
    """One tensor's location inside a safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int


@dataclass
class GroupSummary:
    """What one layer's groups said, summed."""

    groups: int = 0
    identical_scale: int = 0
    same_lattice: int = 0
    off_lattice: int = 0
    codes: int = 0
    flipped: int = 0
    one_apart: int = 0
    scale_drift: list[float] = field(default_factory=list)


@dataclass
class LayerReport:
    """One layer, both sides.

    ``dense`` marks a layer a packed checkpoint left unquantized on either
    side: it holds its source's weights, so it is no lattice's and is reported
    apart from the quantized layers.
    """

    name: str
    width: int
    summary: GroupSummary
    relative_disagreement: float
    elements: int
    differing: int
    dense: bool = False


@dataclass
class PackedLayer:
    """One layer of an MLX checkpoint as it is stored, codes unpacked.

    ``codes`` is shaped ``(out_features, in_features)`` and holds the stored,
    unsigned codes; ``scales`` and ``biases`` are shaped ``(out_features,
    groups)``, widened to float64, which holds bfloat16 exactly.
    """

    codes: np.ndarray
    scales: np.ndarray
    biases: np.ndarray
    bits: int
    group_size: int

    def values(self) -> np.ndarray:
        """``scales * codes + biases``, the weights MLX reconstructs, in float64.

        Exact for MRound's symmetric layers: the bias is minus the stored scale
        times a power of two, so every value is the scale times an integer.
        """
        rows, columns = self.codes.shape
        grouped = self.codes.reshape(rows, -1, self.group_size).astype(np.float64)
        values: np.ndarray = grouped * self.scales[:, :, None] + self.biases[:, :, None]
        return values.reshape(rows, columns)


def read_header(path: Path) -> list[TensorEntry]:
    """The tensors a safetensors file holds, from its header alone.

    The format is eight bytes of little endian header length, the JSON header,
    then the raw buffers; each header entry carries the byte range of its
    tensor relative to the end of the header.
    """
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    base = 8 + length
    entries = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        entries.append(
            TensorEntry(name, spec["dtype"], tuple(spec["shape"]), base + start, base + end)
        )
    return sorted(entries, key=lambda entry: entry.name)


def read_tensor(path: Path, entry: TensorEntry) -> np.ndarray:
    """One tensor, read from its byte range and nothing else, as float64."""
    with path.open("rb") as handle:
        handle.seek(entry.start)
        raw = handle.read(entry.end - entry.start)
    if entry.dtype == "BF16":
        # bfloat16 is the top half of a float32: widen and reinterpret.
        widened = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return widened.view(np.float32).reshape(entry.shape).astype(np.float64)
    dtype = _DTYPES.get(entry.dtype)
    if dtype is None:
        msg = f"{entry.name} is stored as {entry.dtype}, which this reader does not decode"
        raise ValueError(msg)
    return np.frombuffer(raw, dtype=dtype).reshape(entry.shape).astype(np.float64)


def unpack_words(raw: bytes, rows: int, in_features: int, bits: int) -> np.ndarray:
    """The codes of a packed layer, from the bytes of its ``uint32`` words.

    MLX's layout (``mround/formats/packing.py``): code ``i`` of a row occupies
    bits ``[i * bits, (i + 1) * bits)`` of the row read as one little endian
    bit stream, which covers both the even layout of 2, 4 and 8 bits and the
    stream that straddles words at 3, 5 and 6. The words are stored little
    endian, so the stream is the row's bytes in order, least significant bit
    first, on any host.
    """
    per_row = len(raw) // rows if rows else 0
    if rows * per_row != len(raw) or per_row * 8 != in_features * bits:
        msg = (
            f"{len(raw)} bytes over {rows} rows do not hold {in_features} codes of "
            f"{bits} bits a row"
        )
        raise ValueError(msg)
    stream = np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8).reshape(rows, per_row), axis=1, bitorder="little"
    )
    fields = stream.reshape(rows, in_features, bits)
    codes = np.zeros((rows, in_features), dtype=np.int64)
    for place in range(bits):
        codes |= fields[:, :, place].astype(np.int64) << place
    return codes


def packed_layers(entries: list[TensorEntry]) -> set[str]:
    """The block linear weights stored packed: ``uint32`` words with scales beside."""
    names = {entry.name for entry in entries}
    return {
        entry.name
        for entry in linear_weights(entries)
        if entry.dtype == PACKED and entry.name.removesuffix(".weight") + ".scales" in names
    }


def layer_setting(root: Path, path: str) -> tuple[int, int]:
    """A packed layer's ``(bits, group_size)``, from ``config.json``.

    The quantization block is flat: the run's defaults, and a dictionary under
    a layer's module path for each layer that differs from them.
    """
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    block = config.get("quantization")
    if not isinstance(block, dict):
        msg = f"{root} holds packed weights but its config.json has no quantization block"
        raise ValueError(msg)
    override = block.get(path)
    setting = override if isinstance(override, dict) else block
    return int(setting["bits"]), int(setting["group_size"])


def read_packed(root: Path, entries: dict[str, TensorEntry], name: str) -> PackedLayer:
    """One packed layer, its codes unpacked and its scales and biases widened."""
    path = name.removesuffix(".weight")
    bits, group_size = layer_setting(root, path)
    weights = root / "model.safetensors"
    scales = read_tensor(weights, entries[f"{path}.scales"])
    biases = read_tensor(weights, entries[f"{path}.biases"])
    entry = entries[name]
    with weights.open("rb") as handle:
        handle.seek(entry.start)
        raw = handle.read(entry.end - entry.start)
    rows = entry.shape[0]
    codes = unpack_words(raw, rows, scales.shape[1] * group_size, bits)
    return PackedLayer(codes, scales, biases, bits, group_size)


def find_checkpoint(root: Path) -> Path:
    """The directory holding ``config.json``, given it or its parent."""
    if (root / "config.json").is_file():
        return root
    candidates = sorted(p.parent for p in root.glob("*/config.json"))
    if len(candidates) == 1:
        return candidates[0]
    msg = f"{root} holds {len(candidates)} checkpoints; give the one to read"
    raise FileNotFoundError(msg)


def linear_weights(entries: list[TensorEntry]) -> list[TensorEntry]:
    """The block linear weights: two dimensional, inside the layer stack."""
    return [
        entry
        for entry in entries
        if len(entry.shape) == MATRIX
        and ".layers." in entry.name
        and entry.name.endswith(".weight")
    ]


def implied_scales(groups: np.ndarray) -> np.ndarray:
    """Each group's scale, recovered from its values.

    Every value is ``scale`` times an integer, so the scale divides every
    nonzero magnitude and every gap between distinct values; the smallest of
    those is the scale itself whenever the group uses two adjacent codes,
    which a group of 64 weights at 2 to 4 bits does in all but degenerate
    cases. A group of all zeros has no scale and gets ``inf``, so its values
    land at code zero on either side.
    """
    magnitudes = np.where(groups != 0, np.abs(groups), np.inf)
    ordered = np.sort(groups, axis=-1)
    gaps = ordered[..., 1:] - ordered[..., :-1]
    gaps = np.where(gaps > 0, gaps, np.inf)
    scales: np.ndarray = np.minimum(magnitudes.min(axis=-1), gaps.min(axis=-1))
    return scales


def compare_layer(
    name: str,
    left: np.ndarray,
    right: np.ndarray,
    group_size: int,
    *,
    lattice: bool = True,
    stored: tuple[np.ndarray | None, np.ndarray | None] = (None, None),
) -> LayerReport:
    """Codes, lattices and the harness's own disagreement figure for one layer.

    With ``lattice`` off only the float figures are computed, which is what a
    side stored in bfloat16 allows (see ``_COARSE``). ``stored`` gives a side's
    scales when its checkpoint stores them, one per group in the order the
    values are grouped, and they are used in place of the ones its values
    imply.
    """
    if left.shape != right.shape:
        msg = f"{name}: shapes differ, {left.shape} against {right.shape}"
        raise ValueError(msg)
    if left.shape[1] % group_size:
        msg = f"{name}: {left.shape[1]} columns do not divide into groups of {group_size}"
        raise ValueError(msg)
    a = left.reshape(-1, group_size)
    b = right.reshape(-1, group_size)
    if not lattice:
        difference = float(np.sqrt(np.mean((a - b) ** 2)))
        magnitude = float(np.sqrt(np.mean(a * a)))
        return LayerReport(
            name=name,
            width=0,
            summary=GroupSummary(groups=int(a.shape[0])),
            relative_disagreement=difference / max(magnitude, 1e-12),
            elements=int(a.size),
            differing=int((a != b).sum()),
        )
    scale_a, scale_b = (
        implied_scales(values) if known is None else _magnitudes(known)
        for values, known in ((a, stored[0]), (b, stored[1]))
    )

    finite = np.isfinite(scale_a) & np.isfinite(scale_b)
    drift = np.where(finite, np.abs(scale_b / np.where(finite, scale_a, 1.0) - 1.0), 0.0)
    same = finite & (drift <= SAME_LATTICE)
    # Two groups of zeros hold code zero on any lattice, so they count as one.
    # A group with a NaN in it has no scale on its side and counts as neither:
    # its values are not codes of anything.
    zeros = np.isinf(scale_a) & np.isinf(scale_b)
    same |= zeros
    identical = finite & (drift <= IDENTICAL_SCALE)
    identical |= zeros

    summary = GroupSummary(groups=int(a.shape[0]))
    summary.identical_scale = int(identical.sum())
    summary.same_lattice = int(same.sum())
    summary.scale_drift = drift[finite & ~same].tolist()

    # The width is the left side's own: the narrowest symmetric range holding
    # its codes on its own lattice, whatever the right side did. Until
    # 2026-10-02 it was read from the shared groups alone, which left a layer
    # whose two sides share no lattice without one, and the first three pairs
    # that differed in the tuning algorithm, not in a draw, printed three 2
    # bit layers between them as 0 bit.
    own = np.where(np.isfinite(scale_a), scale_a, 1.0)[:, None]
    # Over the groups that have a lattice of their own: a group of zeros has
    # none and holds code zero, and a group with a NaN in it has none either,
    # where its other values read at a scale of one would pass for wide codes.
    lattice_a = np.isfinite(scale_a)
    width = _width(np.rint(a[lattice_a] / own[lattice_a]))

    if summary.same_lattice:
        scale = own[same]
        codes_a = a[same] / scale
        codes_b = b[same] / scale
        nearest_a, nearest_b = np.rint(codes_a), np.rint(codes_b)
        off = (np.abs(codes_a - nearest_a) > INTEGRALITY) | (
            np.abs(codes_b - nearest_b) > INTEGRALITY
        )
        summary.off_lattice = int(off.any(axis=-1).sum())
        moved = np.abs(nearest_a - nearest_b)
        summary.codes = int(codes_a.size)
        summary.flipped = int((moved >= HALF_STEP).sum())
        summary.one_apart = int((np.abs(moved - 1.0) < HALF_STEP).sum())

    difference = float(np.sqrt(np.mean((a - b) ** 2)))
    magnitude = float(np.sqrt(np.mean(a * a)))
    return LayerReport(
        name=name,
        width=width,
        summary=summary,
        relative_disagreement=difference / max(magnitude, 1e-12),
        elements=int(a.size),
        differing=int((a != b).sum()),
    )


def _magnitudes(scales: np.ndarray) -> np.ndarray:
    """Stored scales as the lattice test reads them: magnitudes, a zero as none."""
    magnitudes = np.abs(scales.reshape(-1)).astype(np.float64)
    return np.where(magnitudes > 0, magnitudes, np.inf)


def compare_packed(name: str, left: PackedLayer, right: PackedLayer) -> LayerReport:
    """Two packed layers, compared on what they store.

    A group is on one lattice when its stored scale and bias agree on both
    sides to ``SAME_LATTICE`` of the left scale, and identical when they agree
    to the bit; its stored codes are then compared one by one. A scale that
    changed sign is a different lattice, since its codes mean the opposite
    values. No code is off its lattice, the codes being integers by storage.
    """
    if left.codes.shape != right.codes.shape:
        msg = f"{name}: shapes differ, {left.codes.shape} against {right.codes.shape}"
        raise ValueError(msg)
    if left.group_size != right.group_size:
        msg = f"{name}: groups of {left.group_size} on the left and {right.group_size} on the right"
        raise ValueError(msg)
    scale_a, scale_b = left.scales.reshape(-1), right.scales.reshape(-1)
    bias_a, bias_b = left.biases.reshape(-1), right.biases.reshape(-1)
    identical = (scale_a == scale_b) & (bias_a == bias_b)
    nonzero = scale_a != 0
    drift = np.where(nonzero, np.abs(scale_b / np.where(nonzero, scale_a, 1.0) - 1.0), np.inf)
    same = identical | (
        (drift <= SAME_LATTICE) & (np.abs(bias_b - bias_a) <= SAME_LATTICE * np.abs(scale_a))
    )

    summary = GroupSummary(groups=int(scale_a.size))
    summary.identical_scale = int(identical.sum())
    summary.same_lattice = int(same.sum())
    summary.scale_drift = drift[~same & np.isfinite(drift)].tolist()
    codes_a = left.codes.reshape(-1, left.group_size)[same]
    codes_b = right.codes.reshape(-1, right.group_size)[same]
    moved = np.abs(codes_a - codes_b)
    summary.codes = int(moved.size)
    summary.flipped = int((moved > 0).sum())
    summary.one_apart = int((moved == 1).sum())

    a, b = left.values(), right.values()
    difference = float(np.sqrt(np.mean((a - b) ** 2)))
    magnitude = float(np.sqrt(np.mean(a * a)))
    return LayerReport(
        name=name,
        width=left.bits,
        summary=summary,
        relative_disagreement=difference / max(magnitude, 1e-12),
        elements=int(a.size),
        differing=int((a != b).sum()),
    )


def _width(codes: np.ndarray) -> int:
    """The narrowest symmetric width whose code range holds these codes.

    The range of ``w`` bits is taken as codes of magnitude up to
    ``2 ** (w - 1)`` on either side. The reference clamps its codes to one
    fewer on the positive side and signs its scale, and the scale recovered
    here is a magnitude, so the largest code turns up on either side.

    Read from the finite codes alone. The width comes from every group of a
    layer that has a scale, so one inf among their weights would otherwise
    decide it and would never let an unbounded search end. A NaN does not
    reach here, its group having no scale. The search is bounded as well, so
    that a regression here is a wrong width and not a hang.
    """
    finite = np.abs(codes[np.isfinite(codes)])
    limit = float(finite.max()) if finite.size else 0.0
    width = 2
    while width < WIDEST and 2 ** (width - 1) < limit:
        width += 1
    return width


def coarse_sides(left: Path, right: Path) -> list[str]:
    """Which of the two checkpoints, if either, is stored too coarsely for the lattice.

    A packed checkpoint never is: its quantized layers store their codes, and
    the layers it left dense are reported apart (``LayerReport.dense``).
    """
    coarse = []
    for label, root in (("left", left), ("right", right)):
        entries = read_header(root / "model.safetensors")
        if packed_layers(entries):
            continue
        dtypes = {e.dtype for e in linear_weights(entries)}
        if dtypes & _COARSE:
            coarse.append(f"{label} ({', '.join(sorted(dtypes & _COARSE))})")
    return coarse


def compare(left: Path, right: Path, group_size: int | None = None) -> list[LayerReport]:
    """Every block linear weight of the two checkpoints, one report each.

    ``group_size`` is the fake exports' (``DEFAULT_GROUP_SIZE`` when it is
    ``None``); a packed layer is grouped as its checkpoint says, and a group
    size given against it has to agree.
    """
    left_all = {e.name: e for e in read_header(left / "model.safetensors")}
    right_all = {e.name: e for e in read_header(right / "model.safetensors")}
    left_entries = {e.name: e for e in linear_weights(list(left_all.values()))}
    right_entries = {e.name: e for e in linear_weights(list(right_all.values()))}
    if set(left_entries) != set(right_entries):
        only = sorted(set(left_entries) ^ set(right_entries))
        msg = f"the two checkpoints do not hold the same layers; only on one side: {only[:5]}"
        raise ValueError(msg)
    packed = (packed_layers(list(left_all.values())), packed_layers(list(right_all.values())))
    lattice = not coarse_sides(left, right)
    reports = []
    for name in sorted(left_entries):
        sides: list[PackedLayer | np.ndarray] = []
        for root, entries, own in ((left, left_all, packed[0]), (right, right_all, packed[1])):
            if name in own:
                layer = read_packed(root, entries, name)
                if group_size is not None and group_size != layer.group_size:
                    msg = (
                        f"{root} stores {name} in groups of {layer.group_size}, "
                        f"not the {group_size} given"
                    )
                    raise ValueError(msg)
                sides.append(layer)
            else:
                sides.append(read_tensor(root / "model.safetensors", entries[name]))
        reports.append(_compare_sides(name, sides[0], sides[1], group_size, lattice, packed))
    return reports


def _compare_sides(
    name: str,
    left: PackedLayer | np.ndarray,
    right: PackedLayer | np.ndarray,
    group_size: int | None,
    lattice: bool,
    packed: tuple[set[str], set[str]],
) -> LayerReport:
    """One layer, whatever each side stores it as."""
    if isinstance(left, PackedLayer) and isinstance(right, PackedLayer):
        return compare_packed(name, left, right)
    # A packed checkpoint that stores this layer dense left it unquantized.
    left_dense = bool(packed[0]) and not isinstance(left, PackedLayer)
    right_dense = bool(packed[1]) and not isinstance(right, PackedLayer)
    a = left.values() if isinstance(left, PackedLayer) else left
    b = right.values() if isinstance(right, PackedLayer) else right
    if left_dense or right_dense:
        report = compare_layer(name, a, b, a.shape[1], lattice=False)
        report.dense = True
        return report
    size = next(
        (side.group_size for side in (left, right) if isinstance(side, PackedLayer)),
        group_size if group_size is not None else DEFAULT_GROUP_SIZE,
    )
    stored = tuple(side.scales if isinstance(side, PackedLayer) else None for side in (left, right))
    return compare_layer(name, a, b, size, lattice=lattice, stored=(stored[0], stored[1]))


def same_architecture(left: Path, right: Path) -> str | None:
    """A refusal naming the field, when the two are not one model."""
    a = json.loads((left / "config.json").read_text(encoding="utf-8"))
    b = json.loads((right / "config.json").read_text(encoding="utf-8"))
    for key in _ARCHITECTURE_FIELDS:
        if a.get(key) != b.get(key):
            return f"{key} is {a.get(key)!r} on the left and {b.get(key)!r} on the right"
    return None


def _percent(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.2f}%" if whole else "n/a"


def report(reports: list[LayerReport], coarse: list[str] | None = None) -> None:
    """Print the layer table and the totals.

    ``coarse`` names the side or sides stored in bfloat16 or float16, for
    which only the float figures were computed; the lattice columns and lines
    are then left out rather than printed as zeros that look like a result.
    Layers a packed checkpoint left dense get one line of their own and stay
    out of the table and the means.
    """
    dense = [r for r in reports if r.dense]
    reports = [r for r in reports if not r.dense]
    if dense:
        weights = sum(r.elements for r in dense)
        differing = sum(r.differing for r in dense)
        print(
            f"\n  {len(dense)} layers left dense by a packed side: {differing:,} of "
            f"{weights:,} weights differ ({_percent(differing, weights)})"
        )
    if not reports:
        return
    if coarse:
        _report_floats(reports, coarse)
        return
    heading = ("layer", "bits", "groups", "same", "flipped", "1 apart", "rel")
    print(f"\n{heading[0]:<40} {heading[1]:>4} {heading[2]:>7} {heading[3]:>7} ", end="")
    print(f"{heading[4]:>8} {heading[5]:>8} {heading[6]:>7}")
    for r in reports:
        s = r.summary
        print(
            f"{r.name:<40} {r.width:>4} {s.groups:>7} {_percent(s.same_lattice, s.groups):>7} "
            f"{_percent(s.flipped, s.codes):>8} {_percent(s.one_apart, s.flipped):>8} "
            f"{r.relative_disagreement:>7.4f}"
        )

    by_width: dict[int, GroupSummary] = {}
    total = GroupSummary()
    elements = differing = 0
    for r in reports:
        s = r.summary
        for target in (total, by_width.setdefault(r.width, GroupSummary())):
            target.groups += s.groups
            target.identical_scale += s.identical_scale
            target.same_lattice += s.same_lattice
            target.off_lattice += s.off_lattice
            target.codes += s.codes
            target.flipped += s.flipped
            target.one_apart += s.one_apart
            target.scale_drift.extend(s.scale_drift)
        elements += r.elements
        differing += r.differing

    print(
        f"\n  {len(reports)} layers, {elements:,} weights, {differing:,} differ "
        f"({_percent(differing, elements)})"
    )
    print(
        f"  groups {total.groups:,}: scales identical to {IDENTICAL_SCALE:g} in "
        f"{_percent(total.identical_scale, total.groups)}, on one lattice (to {SAME_LATTICE:g}) in "
        f"{_percent(total.same_lattice, total.groups)}, off lattice {total.off_lattice}"
    )
    if total.scale_drift:
        drift = np.array(total.scale_drift)
        print(
            f"  groups on different lattices {len(drift):,}: scale ratio off one by a median of "
            f"{np.median(drift):.4f}, at most {drift.max():.4f}"
        )
    print(
        f"  codes compared {total.codes:,}: flipped {total.flipped:,} "
        f"({_percent(total.flipped, total.codes)}), of which one step "
        f"{_percent(total.one_apart, total.flipped)}"
    )
    for width in sorted(by_width):
        s = by_width[width]
        layers = sum(1 for r in reports if r.width == width)
        print(
            f"    {width} bit, {layers} layers: flipped {_percent(s.flipped, s.codes)} of "
            f"{s.codes:,} codes, one lattice in {_percent(s.same_lattice, s.groups)} of groups"
        )
    mean = sum(r.relative_disagreement for r in reports) / len(reports)
    print(f"  mean relative disagreement   {mean:.4f}")
    print(SELF_DISAGREEMENT)
    print("  worst layers by codes flipped:")
    for r in sorted(reports, key=lambda r: -r.summary.flipped / max(r.summary.codes, 1))[
        :WORST_SHOWN
    ]:
        print(f"    {_percent(r.summary.flipped, r.summary.codes):>7}  {r.name}")
    print("  worst layers by relative disagreement:")
    for r in sorted(reports, key=lambda r: -r.relative_disagreement)[:WORST_SHOWN]:
        print(f"    {r.relative_disagreement:.4f}  {r.name}")


def _report_floats(reports: list[LayerReport], coarse: list[str]) -> None:
    """The float figures alone, for a pair with a coarsely stored side."""
    print(
        f"\n  lattice, scales and codes not resolvable: {' and '.join(coarse)} stored in a\n"
        "  format whose rounding is comparable to the clipping drift the lattice test\n"
        "  measures, so only the float figures are reported"
    )
    print(f"\n{'layer':<40} {'weights':>11} {'differ':>8} {'rel':>7}")
    for r in reports:
        print(
            f"{r.name:<40} {r.elements:>11,} {_percent(r.differing, r.elements):>8} "
            f"{r.relative_disagreement:>7.4f}"
        )
    elements = sum(r.elements for r in reports)
    differing = sum(r.differing for r in reports)
    mean = sum(r.relative_disagreement for r in reports) / len(reports)
    print(
        f"\n  {len(reports)} layers, {elements:,} weights, {differing:,} differ "
        f"({_percent(differing, elements)}; against a float32 side nearly every value\n"
        "  differs by the rounding alone, so this count is not a count of decisions)"
    )
    print(f"  mean relative disagreement   {mean:.4f}")
    print(SELF_DISAGREEMENT)
    print("  worst layers by relative disagreement:")
    for r in sorted(reports, key=lambda r: -r.relative_disagreement)[:WORST_SHOWN]:
        print(f"    {r.relative_disagreement:.4f}  {r.name}")


def main(argv: list[str] | None = None) -> int:
    """Compare two checkpoint directories and print the report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help=f"the fake exports' group size (default {DEFAULT_GROUP_SIZE}); "
        "a packed checkpoint carries its own",
    )
    args = parser.parse_args(argv)
    try:
        left, right = find_checkpoint(args.left), find_checkpoint(args.right)
    except FileNotFoundError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    refusal = same_architecture(left, right)
    if refusal:
        print(f"refusing: these are two different models: {refusal}", file=sys.stderr)
        return 1
    grouping = (
        f"groups of {args.group_size}"
        if args.group_size is not None
        else f"groups as each packed checkpoint stores them, of {DEFAULT_GROUP_SIZE} otherwise"
    )
    print(f"left   {left}\nright  {right}\n{grouping}")
    try:
        reports = compare(left, right, args.group_size)
    except ValueError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    report(reports, coarse_sides(left, right))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
