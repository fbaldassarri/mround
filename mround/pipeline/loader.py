# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Model loading and architecture detection.

Loading is delegated to ``mlx-lm`` rather than reimplemented. Constructing an
MLX model from a Hugging Face configuration means carrying a per-architecture
model definition for every model MRound wants to quantize, which is a large and
permanently unfinished job that ``mlx-lm`` already does and keeps current. See
MEMORY.md D-018.

The dependency is one-directional and confined to this module and the evaluation
harness. Nothing in `core/`, `reference/`, `planner/`, or `formats/` imports it,
so the quantization mathematics stays independent of whatever ``mlx-lm`` does
next.
"""

from __future__ import annotations

import dataclasses
import platform
from typing import TYPE_CHECKING, Any

from mround.exceptions import ArchitectureError, PlatformError

# MLX is imported lazily, inside the two functions that use it at runtime, and
# this is load-bearing rather than tidiness. `require_apple_silicon` below
# exists to explain the one machine where MLX cannot be installed, a Mac whose
# Python is running under Rosetta. Importing MLX at module scope put that
# explanation inside a module that could not be imported without MLX, so the
# machine it was written for got a ModuleNotFoundError traceback instead. Every
# annotation here is a string under `from __future__ import annotations`, so
# TYPE_CHECKING is enough for them.
if TYPE_CHECKING:
    from pathlib import Path

    import mlx.core as mx
    from mlx import nn

__all__ = ["ModelBundle", "load_model", "model_dtype", "require_apple_silicon"]

_INSTALL_HINT = (
    "Model loading needs the optional model stack. Install it with:\n    pip install -e '.[models]'"
)


@dataclasses.dataclass(frozen=True, slots=True)
class ModelBundle:
    """A loaded model with everything needed to quantize and re-export it.

    Attributes:
        model: The MLX model.
        tokenizer: Its tokenizer, needed to build calibration batches.
        config: The original model configuration, carried through to the
            exported checkpoint so the result loads like the input did.
        source: Where it came from, a Hub repository id or a local path.
        dtype: The model's floating dtype, which the export path uses for
            scales and biases. See DOCUMENTATION.md section 5.1.
    """

    model: nn.Module
    tokenizer: object
    config: dict[str, Any]
    source: str
    dtype: mx.Dtype


def require_apple_silicon() -> None:
    """Raise unless this host can run MRound.

    Checks for macOS on Apple Silicon. MRound targets Apple Silicon only and
    does not fall back to a portable path, so failing early with a clear message
    is better than failing deep inside a tuning loop.

    The processor check is the one that matters in practice. A Python running
    under Rosetta reports ``arm64`` for the machine but ``i386`` for the
    processor, cannot install MLX, and produces a confusing failure much later.

    Raises:
        PlatformError: If the host is not a supported Apple Silicon Mac.
    """
    if platform.system() != "Darwin":
        msg = (
            f"MRound runs on macOS only, and this is {platform.system()}. "
            f"The reference implementation in mround.reference runs anywhere "
            f"and is the right tool for checking the mathematics off-platform."
        )
        raise PlatformError(msg)
    if platform.machine() != "arm64" or platform.processor() != "arm":
        msg = (
            f"MRound needs a native Apple Silicon Python. This one reports "
            f"machine={platform.machine()!r} processor={platform.processor()!r}; "
            f"a processor of 'i386' on an M-series Mac means the interpreter is "
            f"running under Rosetta, which cannot install MLX. See environment.yml."
        )
        raise PlatformError(msg)


def model_dtype(model: nn.Module) -> mx.Dtype:
    """The model's floating dtype.

    Reads it off the parameters rather than the configuration, which may not
    carry it and may disagree with the arrays when it does.

    Skipping non-floating parameters is not defensive tidiness. An
    already-quantized checkpoint stores packed codes as ``uint32`` under the
    name ``weight``, so taking the first parameter's dtype returns ``uint32``
    and every downstream cast is then wrong.

    Raises:
        ArchitectureError: If the model has no floating parameters at all.
    """
    import mlx.core as mx  # noqa: PLC0415
    from mlx.utils import tree_flatten  # noqa: PLC0415

    for _, value in tree_flatten(model.parameters()):
        if mx.issubdtype(value.dtype, mx.floating):
            return value.dtype
    msg = "model has no floating-point parameters, so its dtype cannot be determined"
    raise ArchitectureError(msg)


def load_model(
    source: str | Path, *, dtype: str | None = None, allow_quantized: bool = False
) -> ModelBundle:
    """Load a model from the Hugging Face Hub or a local directory.

    Args:
        source: Hub repository id or local path.
        dtype: Override the stored floating dtype. ``None`` keeps what the
            checkpoint uses, which is what the export path assumes when it
            chooses a scale dtype. See DOCUMENTATION.md section 5.1.
        allow_quantized: Permit loading a checkpoint that is already quantized.
            Off by default because quantizing one again would treat its packed
            codes as weights and produce nonsense without erroring. Evaluation
            needs it on, since scoring a quantized model is the entire point of
            having made one.

    Returns:
        The loaded model and its metadata.

    Raises:
        ArchitectureError: If the architecture is unsupported, or the model is
            already quantized.
        PlatformError: If the host cannot run MRound.
    """
    require_apple_silicon()

    import mlx.core as mx  # noqa: PLC0415

    try:
        from mlx_lm import load  # noqa: PLC0415
    except ImportError as exc:
        raise ArchitectureError(_INSTALL_HINT) from exc

    try:
        model, tokenizer, config = load(str(source), return_config=True)
    except Exception as exc:
        # mlx-lm raises a mixture of ValueError, KeyError, and its own types
        # depending on whether the repository is missing, the architecture is
        # unknown, or the weights do not match. The original is chained for
        # anyone who needs it; what goes in the message is the first line plus
        # the one piece of context the Hub's own error withholds.
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        hint = ""
        if "401" in first_line:
            hint = (
                "\n  A 401 from the Hub usually means the repository does not "
                "exist, not that\n  a token is needed. It answers anonymous "
                "requests about missing and private\n  repositories identically, "
                "and only distinguishes them once you authenticate.\n  Check the "
                "spelling before reaching for a token."
            )
        elif "404" in first_line:
            hint = (
                "\n  The repository does not exist under that name. Model ids "
                "are exact, and\n  suffixes like -bf16 or -mlx are part of the "
                "name rather than a convention."
            )
        msg = f"could not load {source!r}: {first_line}{hint}"
        raise ArchitectureError(msg) from exc

    # Re-quantizing an already-quantized checkpoint would silently produce
    # nonsense: the packed uint32 codes would be treated as weights.
    if not allow_quantized and config.get("quantization") is not None:
        msg = (
            f"{source!r} is already quantized ({config['quantization']}). "
            f"Quantize from the full-precision model instead, or pass "
            f"allow_quantized=True if the intent is to evaluate it."
        )
        raise ArchitectureError(msg)

    if dtype is not None:
        target = getattr(mx, dtype, None)
        if target is None or not isinstance(target, mx.Dtype):
            msg = f"unknown dtype {dtype!r}"
            raise ArchitectureError(msg)
        model.set_dtype(target)

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        config=dict(config),
        source=str(source),
        dtype=model_dtype(model),
    )
