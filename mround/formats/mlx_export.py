# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""MLX-native checkpoint export.

MLX quantized layers reconstruct weights as ``scales * q_u + biases`` where
``q_u`` is the unsigned stored code. Both of MRound's symmetry modes reduce to
that form, arriving by different routes, and the conversion is in
DOCUMENTATION.md section 6.2.

Two things about this format were established by reading MLX's source rather
than its documentation, and both matter here. See MEMORY.md D-017.

**Negative scales are native, not tolerated.** MLX's own affine quantizer
negates the scale for roughly half of all groups, in the same way and for the
same reason MRound does (D-009). The published docstring still shows the
unsigned formula it stopped using. So the signed scale that looked like an
interoperability risk is the format's ordinary case, and nothing downstream
assumes positivity.

**The bias is computed from the rounded scale, never the exact one.** Storing
``scales`` at the model's dtype and then deriving ``biases`` from that stored
value keeps ``scales * 2**(bits-1) + biases`` exactly zero, because multiplying
by a power of two is exact in binary floating point. Derive the bias from the
unrounded scale instead and zero stops being representable, which puts a small
systematic offset on every group in the model.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from mround.exceptions import ExportError, UnsupportedSchemeError
from mround.formats.packing import pack_codes, packed_width
from mround.schemes import QuantScheme, Symmetry

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

__all__ = [
    "MLX_GROUP_SIZES",
    "MLX_SUPPORTED_BITS",
    "QuantizedLayer",
    "export_mlx",
    "require_mlx_representable",
    "to_affine",
]

# What `mx.quantize` accepts. Narrower than MRound's own SUPPORTED_BITS, which
# also covers 7 for the GGUF path: `mx.quantize` rejects 7 explicitly.
MLX_SUPPORTED_BITS: frozenset[int] = frozenset({2, 3, 4, 5, 6, 8})

# Per-channel quantization (group_size -1) has no representation here at all.
MLX_GROUP_SIZES: frozenset[int] = frozenset({32, 64, 128})

# Keys the quantization block uses for its own settings. A module path colliding
# with one of these would be read as a setting and silently ignored as an
# override, so the collision is checked rather than assumed impossible.
RESERVED_CONFIG_KEYS: frozenset[str] = frozenset({"group_size", "bits", "mode"})

GROUPED_NDIM = 3

_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


@dataclasses.dataclass(frozen=True, slots=True)
class QuantizedLayer:
    """One layer's quantization, exactly as :func:`quantize_codes` returns it.

    Attributes:
        codes: Grouped, shaped ``(out_features, n_groups, group_size)``. Signed
            under symmetric schemes, already unsigned under asymmetric ones.
        scale: Shaped ``(out_features, n_groups, 1)``. Signed under symmetric
            schemes, which is correct and not a defect.
        zero_point: Shaped like ``scale`` under asymmetric schemes, ``None``
            under symmetric ones, where the fixed ``2 ** (bits - 1)`` applies.
        scheme: The representation these belong to.
    """

    codes: mx.array
    scale: mx.array
    zero_point: mx.array | None
    scheme: QuantScheme


def _require_mlx_representable(path: str, scheme: QuantScheme) -> None:
    """Refuse a scheme MLX's quantized layers cannot load.

    Public as :func:`require_mlx_representable` so that the entry points can
    ask before a model is loaded and tuned rather than after: a 7 bit or a
    per channel run used to tune every block for an hour and fail here.
    """
    if scheme.bits not in MLX_SUPPORTED_BITS:
        supported = ", ".join(str(b) for b in sorted(MLX_SUPPORTED_BITS))
        msg = (
            f"{path}: MLX quantization accepts bits in {{{supported}}} and "
            f"rejects {scheme.bits}. MRound can pack this width, but no MLX "
            f"consumer can read it, and no other export path exists yet "
            f"(GGUF is ROADMAP.md Phase 4)."
        )
        raise UnsupportedSchemeError(msg)
    if scheme.group_size not in MLX_GROUP_SIZES:
        sizes = ", ".join(str(g) for g in sorted(MLX_GROUP_SIZES))
        detail = "per-channel" if scheme.is_per_channel else str(scheme.group_size)
        msg = (
            f"{path}: MLX quantization accepts group sizes {{{sizes}}} and "
            f"cannot represent {detail}."
        )
        raise UnsupportedSchemeError(msg)


def require_mlx_representable(scheme: QuantScheme, *, path: str = "scheme") -> None:
    """Public form of the check the exporter applies to every layer."""
    _require_mlx_representable(path, scheme)


def to_affine(
    scale: mx.array,
    zero_point: mx.array | None,
    scheme: QuantScheme,
    *,
    dtype: mx.Dtype = mx.bfloat16,
) -> tuple[mx.array, mx.array]:
    """Convert MRound scale parameters to MLX's affine convention.

    Both symmetry modes collapse to ``biases = -scales * zero_point``, with the
    zero point being the fixed integer ``2 ** (bits - 1)`` when the scheme is
    symmetric. Under symmetric schemes that makes the bias exactly the group's
    dominant extreme, which is a real weight value rather than a synthesized
    one, and it is what makes both zero and that extreme reconstruct exactly.

    Args:
        scale: Per-group scale, possibly negative under symmetric schemes.
        zero_point: Per-group zero point, or ``None`` for symmetric schemes.
        scheme: The representation these parameters belong to.
        dtype: The model's floating dtype. ``scales`` and ``biases`` must share
            it, and MLX validates that they share a shape.

    Returns:
        ``(scales, biases)`` satisfying ``W = scales * q_u + biases``.
    """
    scales = scale.astype(dtype)

    # From the stored value, not from `scale`. See the module docstring: this is
    # what keeps zero exactly representable once the scale has been rounded.
    if zero_point is None:
        offset = float(2 ** (scheme.bits - 1))
        biases = (-scales * offset).astype(dtype)
    else:
        biases = (-scales * zero_point.astype(dtype)).astype(dtype)
    return scales, biases


def _tensors_for_layer(path: str, layer: QuantizedLayer, dtype: mx.Dtype) -> dict[str, mx.array]:
    """Packed weight, scales, and biases under the names MLX expects."""
    scheme = layer.scheme
    _require_mlx_representable(path, scheme)

    if layer.codes.ndim != GROUPED_NDIM:
        msg = (
            f"{path}: expected grouped codes shaped (out_features, n_groups, "
            f"group_size), got {layer.codes.shape}"
        )
        raise ExportError(msg)

    out_features, n_groups, group_size = layer.codes.shape
    if group_size != scheme.group_size:
        msg = f"{path}: codes are grouped by {group_size} but the scheme says {scheme.group_size}"
        raise ExportError(msg)

    # Symmetric codes are signed and storage is unsigned, so the fixed offset
    # goes on here rather than being left to the caller to remember.
    flat = layer.codes.reshape(out_features, n_groups * group_size)
    if scheme.symmetry is Symmetry.SYMMETRIC:
        flat = flat + 2 ** (scheme.bits - 1)

    packed = pack_codes(flat.astype(mx.uint32), scheme.bits)
    expected = packed_width(n_groups * group_size, scheme.bits)
    if packed.shape != (out_features, expected):
        msg = f"{path}: packed to {packed.shape}, expected {(out_features, expected)}"
        raise ExportError(msg)

    scales, biases = to_affine(layer.scale, layer.zero_point, scheme, dtype=dtype)
    flat_shape = (out_features, n_groups)
    return {
        f"{path}.weight": packed,
        f"{path}.scales": scales.reshape(flat_shape),
        f"{path}.biases": biases.reshape(flat_shape),
    }


def _quantization_block(quantized: Mapping[str, QuantizedLayer]) -> dict[str, Any]:
    """The config.json quantization block, with per-layer overrides.

    The schema is flat rather than nested: overrides live in the same dictionary
    as the defaults, keyed by dotted module path. That is MLX's design, not a
    simplification made here, and it is why the reserved-key collision below is
    a real possibility worth checking rather than a theoretical one.
    """
    schemes = [layer.scheme for layer in quantized.values()]
    default = max(set(schemes), key=schemes.count)

    block: dict[str, Any] = {
        "group_size": default.group_size,
        "bits": default.bits,
        "mode": "affine",
    }
    for path, layer in sorted(quantized.items()):
        if layer.scheme == default:
            continue
        if path in RESERVED_CONFIG_KEYS:
            msg = (
                f"module path {path!r} collides with a quantization setting key, "
                f"so its per-layer override would be silently ignored"
            )
            raise ExportError(msg)
        block[path] = {
            "group_size": layer.scheme.group_size,
            "bits": layer.scheme.bits,
            "mode": "affine",
        }
    return block


def export_mlx(
    output_dir: Path,
    dense_weights: Mapping[str, mx.array],
    quantized: Mapping[str, QuantizedLayer],
    config: Mapping[str, Any],
    *,
    dtype: str = "bfloat16",
    extra_files: Mapping[str, str] | None = None,
) -> None:
    """Write a checkpoint that ``mlx-lm`` loads without modification.

    Per-layer bit widths and group sizes are recorded in the configuration so
    that mixed-precision models load correctly. A checkpoint that quietly
    assumes a uniform scheme would load and then generate nonsense.

    Sharding is not implemented. mlx-lm splits at 5 GB per file; this writes one
    file whatever the size, which loads the same way but is a deviation from
    that convention whose effect on hub tooling is untested.

    Args:
        output_dir: Destination, created if absent.
        dense_weights: Tensors kept at full precision, written verbatim. Layers
            appearing here and not in ``quantized`` stay dense, which mlx-lm
            supports: it decides per layer on whether ``<path>.scales`` exists.
        quantized: Per-layer quantization, keyed by module path with no
            ``.weight`` suffix.
        config: The source model's configuration. Copied, then extended.
        dtype: The model's floating dtype, shared by scales and biases.
        extra_files: Filename to text content, written alongside. Tokenizer
            files go here, so the result is self-contained.

    Raises:
        ExportError: If the destination cannot be written, or a layer's tensors
            are shaped inconsistently with its scheme.
        UnsupportedSchemeError: If a layer's scheme has no MLX representation.
    """
    if dtype not in _DTYPES:
        known = ", ".join(sorted(_DTYPES))
        msg = f"dtype must be one of {{{known}}}, got {dtype!r}"
        raise ExportError(msg)
    if not quantized:
        msg = "nothing to export: no quantized layers were supplied"
        raise ExportError(msg)

    target = _DTYPES[dtype]
    tensors: dict[str, mx.array] = dict(dense_weights)
    for path, layer in sorted(quantized.items()):
        collision = f"{path}.weight"
        if collision in dense_weights:
            msg = f"{collision} appears as both a dense and a quantized tensor"
            raise ExportError(msg)
        tensors.update(_tensors_for_layer(path, layer, target))

    merged = dict(config)
    block = _quantization_block(quantized)
    merged["quantization"] = block
    # Duplicated under a second key, which is what mlx-lm writes for
    # compatibility with tooling that reads the Hugging Face convention.
    merged["quantization_config"] = block

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_file = output_dir / "model.safetensors"
        mx.save_safetensors(str(weights_file), tensors, metadata={"format": "mlx"})
        (output_dir / "config.json").write_text(json.dumps(merged, indent=2) + "\n")
        # The two metadata fields follow mlx-lm's own writer: total_size is
        # the byte count of every stored tensor, and total_parameters counts
        # the weights a tensor represents rather than the words it is packed
        # into, so a 4 bit layer counts each uint32 word as eight parameters
        # and its scales and biases are not parameters at all. Counting
        # stored elements instead reported about an eighth of the truth at 4
        # bits, in the field Hub tooling reads for model size.
        packed_names = {f"{path}.weight" for path in quantized}
        metadata_names = {f"{path}.{kind}" for path in quantized for kind in ("scales", "biases")}
        total_parameters = sum(
            t.size * 32 // quantized[name.removesuffix(".weight")].scheme.bits
            if name in packed_names
            else t.size
            for name, t in tensors.items()
            if name not in metadata_names
        )
        index = {
            "metadata": {
                "total_size": sum(t.nbytes for t in tensors.values()),
                "total_parameters": total_parameters,
            },
            "weight_map": dict.fromkeys(sorted(tensors), "model.safetensors"),
        }
        (output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        for name, content in (extra_files or {}).items():
            (output_dir / name).write_text(content)
    except OSError as exc:
        msg = f"could not write the checkpoint to {output_dir}: {exc}"
        raise ExportError(msg) from exc
