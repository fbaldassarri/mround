# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Perplexity evaluation.

The primary quality measure. Absolute values are only comparable within a fixed
evaluation set and stride, so every result carries the settings that produced
it; a perplexity quoted without them is not a number anyone can check.

Three implementation details change the answer and are therefore decided here
rather than inherited:

**Logits are cast to float32 before the loss.** A cross entropy over a hundred
thousand vocabulary entries in bfloat16 loses real precision, and the whole
point of this measurement is to resolve small differences between a quantized
model and its original.

**The graph is evaluated once per batch.** MLX is lazy, so without that the
losses accumulate into one unevaluated graph spanning the entire evaluation set
and memory grows until it does not.

**Overlapping windows score only their new tokens.** With a stride shorter than
the context window, every token would otherwise be scored several times, once
with little context and again with more, which quietly lowers the result. Only
the tokens a window sees for the first time contribute.

MLX is imported inside the one function that needs it rather than at module
scope. That is not a style preference: it makes the window schedule and the
dataset handling importable and testable on a machine with no MLX, and the
window schedule is the part of this file most likely to be quietly wrong.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING

from mround.exceptions import CalibrationError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mlx import nn

__all__ = ["PerplexityResult", "evaluate_perplexity", "load_evaluation_text"]

# Where a short name resolves to on the Hub, with the configuration and the
# split to score. Repository ids are namespaced, which is not cosmetic: the
# `datasets` library rejected bare names in version 5, and `wikitext` moved
# under an organization at some point before that, so the unqualified form was
# already a redirect waiting to expire.
KNOWN_DATASETS: dict[str, tuple[str, str, str]] = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "test"),
}


@dataclasses.dataclass(frozen=True, slots=True)
class PerplexityResult:
    """A perplexity measurement with everything needed to reproduce it.

    Attributes:
        perplexity: The measurement.
        n_tokens: Tokens scored.
        dataset: Evaluation set identifier.
        seq_len: Context window used.
        stride: Sliding window stride. Equal to ``seq_len`` means
            non-overlapping windows; smaller values score each token with more
            context and lower the result, which is why it must be reported.
    """

    perplexity: float
    n_tokens: int
    dataset: str
    seq_len: int
    stride: int

    def describe(self) -> str:
        """One line carrying the settings, so the number stays checkable."""
        return (
            f"perplexity {self.perplexity:.4f} on {self.dataset} "
            f"({self.n_tokens} tokens, seq_len {self.seq_len}, stride {self.stride})"
        )


def load_evaluation_text(dataset: str) -> str:
    """Resolve a dataset identifier to text.

    A local path is read directly, which is what makes this testable and what
    lets a run be pinned to an exact file rather than to whatever a Hub dataset
    contains today.

    Args:
        dataset: A known identifier from :data:`KNOWN_DATASETS`, or a path to a
            UTF-8 text file.

    Returns:
        The concatenated text.

    Raises:
        CalibrationError: If the identifier is neither known nor a readable
            file, or if the dataset stack is not installed.
    """
    path = Path(dataset)
    if path.is_file():
        return path.read_text(encoding="utf-8")

    if dataset not in KNOWN_DATASETS:
        known = ", ".join(sorted(KNOWN_DATASETS))
        msg = f"{dataset!r} is neither a readable file nor one of {{{known}}}"
        raise CalibrationError(msg)

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            f"scoring {dataset!r} needs the evaluation stack. Install it with:\n"
            f"    pip install -e '.[eval]'\n"
            f"Or pass a path to a local text file instead."
        )
        raise CalibrationError(msg) from exc

    name, subset, split = KNOWN_DATASETS[dataset]
    try:
        rows = load_dataset(name, subset, split=split)
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        msg = (
            f"could not fetch {name}/{subset}[{split}]: {first_line}\n"
            f"  Dataset ids move and the `datasets` library has tightened what it "
            f"accepts more\n  than once. Passing a local text file avoids the "
            f"question entirely and pins the\n  evaluation to an exact file, which "
            f"a published number needs anyway."
        )
        raise CalibrationError(msg) from exc
    return "\n\n".join(rows["text"])


def _encode_prefix(tokenizer: object, text: str, needed: int | None) -> list[int]:
    """Tokenize only as much text as the requested token count needs.

    Encoding a whole evaluation set to score a few thousand tokens of it is
    wasteful, and it also triggers a warning from the tokenizer about the
    sequence exceeding the model's context. That warning is a false alarm here,
    since the tokens are windowed before they reach the model, but it looks
    exactly like a real problem and appears on every run.

    Growing the slice rather than guessing a bytes-per-token ratio keeps this
    correct for any tokenizer and any language. Byte-level tokenizers on dense
    scripts can fall well below the four characters per token that English
    suggests, and a wrong guess here would silently truncate the evaluation.
    """
    encode = tokenizer.encode  # type: ignore[attr-defined]
    if needed is None:
        return list(encode(text))

    chars = needed * 8
    while chars < len(text):
        ids = list(encode(text[:chars]))
        if len(ids) >= needed:
            return ids[:needed]
        chars *= 2
    return list(encode(text))[:needed]


def _windows(n_tokens: int, seq_len: int, stride: int) -> Iterator[tuple[int, int]]:
    """Yield ``(start, n_scored)`` for each window.

    ``n_scored`` counts only the targets this window sees first, so overlapping
    windows do not score the same token twice.
    """
    start = 0
    scored_through = 1  # position 0 is never a target; it is the first input
    while start + 1 < n_tokens:
        end = min(start + seq_len, n_tokens)
        n_scored = end - max(scored_through, start + 1)
        if n_scored > 0:
            yield start, n_scored
            scored_through = end
        if end >= n_tokens:
            break
        start += stride


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: object,
    *,
    dataset: str = "wikitext2",
    seq_len: int = 2048,
    stride: int | None = None,
    max_tokens: int | None = None,
) -> PerplexityResult:
    """Measure perplexity on a held-out set.

    Args:
        model: A loaded MLX model, quantized or not.
        tokenizer: Its tokenizer.
        dataset: Evaluation set identifier, or a path to a text file.
        seq_len: Context window.
        stride: Sliding window stride. ``None`` uses non-overlapping windows.
        max_tokens: Stop after this many tokens, for quick checks. ``None``
            scores the whole set, which is what a published number requires.

    Returns:
        The measurement and its settings.

    Raises:
        CalibrationError: If the evaluation set cannot be resolved, or is too
            short to score even one window.
    """
    import mlx.core as mx  # noqa: PLC0415
    from mlx import nn as nn_  # noqa: PLC0415

    stride = seq_len if stride is None else stride
    if not 0 < stride <= seq_len:
        msg = f"stride must be in (0, seq_len]; got stride={stride}, seq_len={seq_len}"
        raise CalibrationError(msg)

    text = load_evaluation_text(dataset)
    ids = _encode_prefix(tokenizer, text, None if max_tokens is None else max_tokens + 1)
    if len(ids) < 2:  # noqa: PLR2004
        msg = f"{dataset!r} tokenized to {len(ids)} tokens, which scores nothing"
        raise CalibrationError(msg)

    tokens = mx.array(ids)
    total_nll = 0.0
    total_scored = 0

    for start, n_scored in _windows(len(ids), seq_len, stride):
        window = tokens[None, start : start + seq_len]
        logits = model(window[:, :-1]).astype(mx.float32)
        losses = nn_.losses.cross_entropy(logits, window[:, 1:], reduction="none")

        # Only the tail this window is the first to see. For non-overlapping
        # windows that is all of them and the slice is a no-op.
        scored = losses[:, -n_scored:]
        batch_nll = mx.sum(scored)
        mx.eval(batch_nll)

        total_nll += float(batch_nll)
        total_scored += n_scored

    mean_nll = total_nll / total_scored
    return PerplexityResult(
        perplexity=math.exp(mean_nll),
        n_tokens=total_scored,
        dataset=dataset,
        seq_len=seq_len,
        stride=stride,
    )
