# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Integer code packing into unsigned 32-bit words, in NumPy.

Implements DOCUMENTATION.md section 6.1. Two layouts, chosen by whether the bit
width divides 32 evenly, with element zero always in the least significant
field.

Packing is cheap to test exhaustively and expensive to get wrong silently, which
is why the round-trip property is verified over random data at every supported
width rather than by inspection. A corrupted packing produces a checkpoint that
loads without complaint and generates nonsense.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mround.schemes import SUPPORTED_BITS

__all__ = ["pack_codes", "packed_width", "unpack_codes"]

Codes = npt.NDArray[np.integer]
Packed = npt.NDArray[np.uint32]

WORD_BITS = 32
MATRIX_NDIM = 2


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_BITS:
        supported = ", ".join(str(b) for b in sorted(SUPPORTED_BITS))
        msg = f"bits must be one of {{{supported}}}, got {bits}"
        raise ValueError(msg)


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


def pack_codes(codes: Codes, bits: int) -> Packed:
    """Pack unsigned integer codes into uint32 words.

    Args:
        codes: Shaped ``(out_features, in_features)``, already offset into the
            unsigned range. Symmetric schemes apply their fixed
            ``2 ** (bits - 1)`` offset before calling this.
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
    if codes.min(initial=0) < 0 or codes.max(initial=0) > hi:
        msg = f"codes must lie in [0, {hi}] at {bits} bits, got [{codes.min()}, {codes.max()}]"
        raise ValueError(msg)

    rows, cols = codes.shape
    words = packed_width(cols, bits)
    wide = codes.astype(np.uint64)

    if WORD_BITS % bits == 0:
        per_word = WORD_BITS // bits
        blocks = wide.reshape(rows, words, per_word)
        shifts = (np.arange(per_word, dtype=np.uint64) * np.uint64(bits)).reshape(1, 1, -1)
        packed: npt.NDArray[np.uint64] = np.bitwise_or.reduce(blocks << shifts, axis=-1)
        return packed.astype(np.uint32)

    # Cross-word bitstream: element i occupies absolute bits [i*bits,
    # (i+1)*bits) of a little-endian word stream, and may straddle a boundary.
    # Exactly 32 codes occupy exactly `bits` words.
    out = np.zeros((rows, words), dtype=np.uint64)
    for i in range(cols):
        start = i * bits
        word, offset = divmod(start, WORD_BITS)
        out[:, word] |= wide[:, i] << np.uint64(offset)
        spill = offset + bits - WORD_BITS
        if spill > 0:
            out[:, word + 1] |= wide[:, i] >> np.uint64(bits - spill)
    return (out & np.uint64(0xFFFFFFFF)).astype(np.uint32)


def unpack_codes(packed: Packed, in_features: int, bits: int) -> Codes:
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
    mask = np.uint64(2**bits - 1)
    wide = packed.astype(np.uint64)

    if WORD_BITS % bits == 0:
        per_word = WORD_BITS // bits
        shifts = (np.arange(per_word, dtype=np.uint64) * np.uint64(bits)).reshape(1, 1, -1)
        codes = (wide[:, :, None] >> shifts) & mask
        return codes.reshape(rows, -1)[:, :in_features].astype(np.int64)

    out = np.zeros((rows, in_features), dtype=np.uint64)
    for i in range(in_features):
        start = i * bits
        word, offset = divmod(start, WORD_BITS)
        value = wide[:, word] >> np.uint64(offset)
        spill = offset + bits - WORD_BITS
        if spill > 0:
            value |= wide[:, word + 1] << np.uint64(bits - spill)
        out[:, i] = value & mask
    return out.astype(np.int64)
