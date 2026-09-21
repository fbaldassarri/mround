# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""GGUF export for llama.cpp.

Maps MRound schemes onto GGUF quantization types where a faithful mapping
exists, and fails where none does. A quantizer that quietly produces something
other than what was requested is worse than one that refuses, so approximating a
scheme into the nearest GGUF type is not an option this module offers.

The signed-scale case is the first place that rule bites. GGUF block formats
assume a particular scale convention, and a group whose scale is negative must
be normalized into it by negating the codes within the group, which is exact,
rather than by taking an absolute value, which corrupts the weights.

Status: Phase 4. Not implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import mlx.core as mx

    from mround.schemes import QuantScheme

__all__ = ["export_gguf", "gguf_type_for"]


def gguf_type_for(scheme: QuantScheme) -> str:
    """Return the GGUF quantization type that represents ``scheme`` exactly.

    Args:
        scheme: The representation to map.

    Returns:
        The GGUF type name.

    Raises:
        UnsupportedSchemeError: If no GGUF type represents ``scheme`` without
            loss. The message names both the scheme and the closest types, so
            the caller can choose one deliberately rather than discovering the
            substitution later.
    """
    raise NotImplementedError


def export_gguf(
    output_path: Path,
    *,
    weights: dict[str, mx.array],
    scale_by_layer: dict[str, mx.array],
    zero_point_by_layer: dict[str, mx.array | None],
    scheme_by_layer: dict[str, QuantScheme],
    config: dict[str, object],
    tokenizer: object,
) -> None:
    """Write a GGUF file that ``llama.cpp`` loads without modification.

    Six same-shaped mapping arguments invite positional transposition, so
    everything after the destination is keyword-only.

    Args:
        output_path: Destination file.
        weights: Every tensor, quantized and unquantized alike.
        scale_by_layer: Per-group scales for quantized layers.
        zero_point_by_layer: Per-group zero points, ``None`` where symmetric.
        scheme_by_layer: The scheme each quantized layer used.
        config: The source model's configuration, translated to GGUF metadata.
        tokenizer: Required, since GGUF embeds the vocabulary.

    Raises:
        ExportError: If the destination cannot be written.
        UnsupportedSchemeError: If any layer's scheme has no faithful GGUF type.
    """
    raise NotImplementedError
