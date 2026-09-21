# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Integer code packing into unsigned 32-bit words.

Codes pack along the input dimension with element zero in the least significant
field. Two layouts, chosen by whether the bit width divides 32:

- Evenly (2, 4, 8): each word holds ``32 // bits`` codes at fixed offsets.
- Otherwise (3, 5, 6, 7): codes form a contiguous little-endian bitstream in
  which element ``i`` occupies absolute bits ``[i*b, (i+1)*b)`` and may straddle
  a word boundary. Exactly 32 codes occupy exactly ``b`` words.

Packing is cheap to test exhaustively and expensive to get wrong silently, which
is why the round-trip property is verified over random data at every supported
width rather than by inspection.

This layout is not MRound's invention and must not drift from MLX's, since a
checkpoint that packs differently loads without complaint and generates
nonsense. `tests/unit/test_reference_packing.py` implements MLX's documented
algorithm independently and asserts the two agree at every width; MEMORY.md
D-017 records why that test exists.

The two branches produce identical bytes, so the even one is purely an
optimization. It is kept because the general branch expands the data by a factor
of ``bits`` before folding it back down, and 2, 4, and 8 are the widths this
project exists to serve (D-007).

This lives in the formats layer rather than the kernels layer because it is part
of a checkpoint's definition, not part of the compute path. A Metal kernel that
accelerates it would live in :mod:`mround.kernels` and be called from here.

Specification: DOCUMENTATION.md section 6.1.
"""

from __future__ import annotations

import mlx.core as mx

from mround.schemes import SUPPORTED_BITS

__all__ = ["pack_codes", "packed_width", "unpack_codes"]

WORD_BITS = 32
MATRIX_NDIM = 2


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS:
        supported = ", ".join(str(b) for b in sorted(SUPPORTED_BITS))
        msg = f"bits must be one of {{{supported}}}, got {bits}"
        raise ValueError(msg)


def _fold_or(placed: mx.array) -> mx.array:
    """Combine the last axis with bitwise or, halving until one field remains.

    MLX has no bitwise-or reduction, so this stands in for the reference's
    ``np.bitwise_or.reduce``. Halving needs an even axis at every step, which
    means a power of two overall, and the axis is not always one: the bitstream
    unpack folds ``bits`` fields, and five and seven are not powers of two. The
    shortfall is padded with zeros, which is exact because zero is the identity
    for or.

    That padding is not defensive tidiness. Without it the halving either raises
    on an unbroadcastable pair, or silently duplicates a field and gets away with
    it because or is idempotent. The second is the dangerous one, so the axis is
    made a power of two rather than left to luck.

    A sum would give the same answer here, because the fields being combined are
    disjoint by construction. Or is used anyway: it says what is meant, and it
    cannot be perturbed by whatever accumulation dtype a reduction picks.
    """
    fields = placed.shape[-1]
    target = 1 << (fields - 1).bit_length()
    if target > fields:
        pad = mx.zeros((*placed.shape[:-1], target - fields), dtype=placed.dtype)
        placed = mx.concatenate([placed, pad], axis=-1)
    while placed.shape[-1] > 1:
        half = placed.shape[-1] // 2
        placed = placed[..., :half] | placed[..., half:]
    return placed[..., 0]


def packed_width(in_features: int, bits: int) -> int:
    """Number of uint32 words one output row occupies.

    Args:
        in_features: Codes per row.
        bits: Bit width.

    Returns:
        Words per row.

    Raises:
        ValueError: If ``in_features`` does not tile evenly under ``bits``. The
            even layout needs a multiple of ``32 // bits``; the bitstream layout
            needs a multiple of 32.
    """
    _validate_bits(bits)
    if WORD_BITS % bits == 0:
        per_word = WORD_BITS // bits
        if in_features % per_word:
            msg = (
                f"in_features={in_features} is not a multiple of {per_word} "
                f"codes per word at {bits} bits"
            )
            raise ValueError(msg)
        return in_features // per_word

    if in_features % WORD_BITS:
        msg = (
            f"in_features={in_features} is not a multiple of 32, which the "
            f"cross-word bitstream layout requires at {bits} bits"
        )
        raise ValueError(msg)
    return in_features // WORD_BITS * bits


def pack_codes(codes: mx.array, bits: int) -> mx.array:
    """Pack unsigned integer codes into uint32 words.

    Args:
        codes: Shaped ``(out_features, in_features)``, already offset into the
            unsigned range. Symmetric schemes apply their fixed
            ``2 ** (bits - 1)`` offset before calling this; see
            DOCUMENTATION.md section 6.2.
        bits: Bit width.

    Returns:
        Shaped ``(out_features, packed_width(in_features, bits))``.

    Raises:
        ValueError: If any code falls outside ``[0, 2**bits - 1]``, or if the
            row length does not tile. Both indicate a bug upstream, and masking
            the codes instead would corrupt weights silently.
    """
    _validate_bits(bits)
    if codes.ndim != MATRIX_NDIM:
        msg = f"expected a 2-D code matrix, got shape {codes.shape}"
        raise ValueError(msg)

    hi = 2**bits - 1
    lo_seen, hi_seen = int(codes.min()), int(codes.max())
    if lo_seen < 0 or hi_seen > hi:
        msg = f"codes must lie in [0, {hi}] at {bits} bits, got [{lo_seen}, {hi_seen}]"
        raise ValueError(msg)

    rows, cols = codes.shape
    words = packed_width(cols, bits)
    wide = codes.astype(mx.uint32)

    if WORD_BITS % bits == 0:
        per_word = WORD_BITS // bits
        blocks = wide.reshape(rows, words, per_word)
        shifts = mx.arange(per_word, dtype=mx.uint32) * bits
        return _fold_or(blocks << shifts)

    # Explode to individual bits, least significant first, regroup into words of
    # 32, and reassemble. Element i then lands on absolute bits [i*b, (i+1)*b)
    # of a little-endian stream, straddling word boundaries where it must.
    exploded = (wide[:, :, None] >> mx.arange(bits, dtype=mx.uint32)) & 1
    grouped = exploded.reshape(rows, words, WORD_BITS)
    return _fold_or(grouped << mx.arange(WORD_BITS, dtype=mx.uint32))


def unpack_codes(packed: mx.array, in_features: int, bits: int) -> mx.array:
    """Recover integer codes from packed words.

    Exactly inverts :func:`pack_codes`. The pair is the round-trip property the
    unit tests exercise over random data at every supported width.

    Args:
        packed: Shaped ``(out_features, packed_width(in_features, bits))``.
        in_features: Codes per row, needed because trailing bits in the final
            word are not self-describing.
        bits: Bit width.

    Returns:
        Shaped ``(out_features, in_features)``.

    Raises:
        ValueError: If the word count does not match what ``in_features`` and
            ``bits`` imply.
    """
    _validate_bits(bits)
    expected = packed_width(in_features, bits)
    if packed.shape[1] != expected:
        msg = (
            f"expected {expected} words per row for in_features={in_features} "
            f"at {bits} bits, got {packed.shape[1]}"
        )
        raise ValueError(msg)

    rows = packed.shape[0]
    mask = 2**bits - 1
    wide = packed.astype(mx.uint32)

    if WORD_BITS % bits == 0:
        per_word = WORD_BITS // bits
        shifts = mx.arange(per_word, dtype=mx.uint32) * bits
        codes = (wide[:, :, None] >> shifts) & mask
        return codes.reshape(rows, -1)[:, :in_features]

    stream = ((wide[:, :, None] >> mx.arange(WORD_BITS, dtype=mx.uint32)) & 1).reshape(rows, -1)
    grouped = stream[:, : in_features * bits].reshape(rows, in_features, bits)
    return _fold_or(grouped << mx.arange(bits, dtype=mx.uint32))
