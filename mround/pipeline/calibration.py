# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Calibration data handling.

A moving calibration set invalidates every comparison a project like this makes,
so the corpus is frozen early and identified by a content hash that travels with
every result. Two numbers quoting the same hash were measured on the same tokens.
Two quoting different hashes are not comparable, and the hash says so before
anyone spends an afternoon wondering why a change helped.

Four choices here change the result and are therefore decided rather than
inherited from whatever the loader happened to return.

**The default matches the reference implementation.** Documents shorter than the
sequence length are dropped and the survivors are truncated to it. That is not
the corpus this file would build if comparability were free: at 2048 tokens it
keeps only the long documents, which is a real selection on a mixed corpus. It is
what the reference does by default, and MEMORY.md Q-003 decided that matching it
is worth more right now than a cleaner corpus of our own, because a difference
between MRound and the reference should be attributable to the method rather than
to the data.

**Concatenation is available and off, exactly as in the reference.** It packs
documents end to end and chops the stream into full-length sequences, which uses
the short-form text the default discards and keeps no document over any other.
Selecting between the two is a measurement, not an opinion, and the content hash
makes the two corpora tell themselves apart.

**One quality filter is applied in both modes.** A sequence whose final token
occupies more than half of it is dropped. That screens out whitespace runs and
padding-like documents, which contribute a strong and entirely uninformative
signal to a reconstruction loss. The rule and the threshold come from the
reference.

**Nothing here is random by accident.** The document order comes from an explicit
seed, and the seed goes into the content hash, so the same arguments always
produce the same corpus and a different seed is visible as a different hash
rather than as unexplained drift.

Two defects in the reference's own loader are deliberately not reproduced, per
Q-003. Its concatenation path leaks a special-token counter across documents,
which eventually emits sequences of the wrong length and can empty the corpus
entirely; the implementation here reserves that space per sequence, where it
belongs. Its ultrachat path overwrites the chat-template decision unconditionally
one line after making it. Neither is behavior worth matching, and both are
recorded rather than silently fixed.

MLX is imported inside the functions that need arrays rather than at module
scope, which keeps the selection, the packing, and the hashing importable and
testable on a machine with no MLX. Those are the parts worth testing.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import inspect
import random
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mround.exceptions import CalibrationError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    import mlx.core as mx

__all__ = [
    "KNOWN_CORPORA",
    "CalibrationSet",
    "Packing",
    "build_calibration_set",
    "content_hash",
    "load_calibration_set",
    "save_calibration_set",
]


class Packing(enum.StrEnum):
    """How a corpus of documents becomes fixed-length sequences.

    Attributes:
        FILTER: Keep documents already at least ``seq_len`` tokens and truncate
            them to it. The reference implementation's default, and therefore
            MRound's.
        CONCAT: Join documents end to end and chop the stream into full-length
            sequences. Uses the short-form text ``FILTER`` discards, at the cost
            of sequences that span a document boundary.
    """

    FILTER = "filter"
    CONCAT = "concat"


# Where a short name resolves to on the Hub, as (repository, configuration,
# split, text column). Every entry has been checked against the Hub's dataset
# index rather than assumed: a wrong column name fails late, and a wrong split
# calibrates on data the model was trained on.
KNOWN_CORPORA: dict[str, tuple[str, str | None, str, str]] = {
    "pile-10k": ("NeelNanda/pile-10k", None, "train", "text"),
    "wiki-10k": ("NeelNanda/wiki-10k", None, "train", "text"),
}

# The default corpus, and the reference's. Ten thousand documents sampled from
# the Pile, which is mixed-domain by construction: prose, code, dialogue, and
# markup in one place. Calibration on a single domain tunes the rounding for that
# domain, and the damage does not show up until the model is used for something
# else.
DEFAULT_CORPUS = "pile-10k"

# The reference's seed. Arbitrary, as any seed is, but not freshly arbitrary:
# reusing it leaves one fewer difference to argue about when MRound's numbers are
# put beside the reference's.
DEFAULT_SEED = 42

# Bumped whenever the sequence building changes in a way that produces different
# tokens from the same arguments. It is part of the hash so that a corpus saved
# by an older version cannot silently pass for a current one.
_HASH_VERSION = "mround-calibration-v1"

# A sequence in which one token occupies more than this fraction is degenerate.
# Two, meaning more than half, is the reference's threshold.
_REPETITION_DIVISOR = 2

# Below this the repetition rule stops being meaningful, since almost any short
# sequence has some token in half its positions. The reference guards the same way.
_MIN_SEQ_LEN_FOR_REPETITION = 2


@dataclasses.dataclass(frozen=True, slots=True)
class CalibrationSet:
    """Tokenized calibration data, with the provenance to reproduce it.

    Attributes:
        batches: Tokenized sequences, already batched. Each is
            ``(batch, seq_len)`` of token ids.
        attention_masks: Per-batch masks, or ``None`` when every sequence is
            full length and nothing is padded. Both packing modes produce
            ``None``; the field exists for a future builder that cannot.
        source: Dataset identifier the corpus came from.
        packing: How documents became sequences.
        n_samples: Sequences in total.
        seq_len: Tokens per sequence.
        seed: Seed that fixed the document order.
        content_hash: Hash of the tokenized content and the settings that
            produced it. Two runs quoting the same hash used the same data,
            which is what makes their numbers comparable.
    """

    batches: list[mx.array]
    attention_masks: list[mx.array] | None
    source: str
    packing: Packing
    n_samples: int
    seq_len: int
    seed: int
    content_hash: str

    def __len__(self) -> int:
        """Number of batches."""
        return len(self.batches)

    @property
    def batch_size(self) -> int:
        """Sequences per batch, from the first batch."""
        return int(self.batches[0].shape[0])

    def describe(self) -> str:
        """One line carrying the provenance, so a result stays checkable."""
        return (
            f"{self.n_samples} x {self.seq_len} tokens from {self.source} "
            f"({self.packing}, seed {self.seed}, {len(self.batches)} batches, "
            f"hash {self.content_hash})"
        )


def _is_degenerate(sample: Sequence[int], seq_len: int) -> bool:
    """Whether one token occupies more than half the sequence.

    Screens out whitespace runs, padding-like documents, and repeated boilerplate.
    These are not rare in a web-scraped corpus and they are actively harmful as
    calibration: a reconstruction loss dominated by a token that means nothing
    tunes the rounding to reproduce nothing accurately.
    """
    return (
        len(sample) > 1
        and seq_len > _MIN_SEQ_LEN_FOR_REPETITION
        and list(sample).count(sample[-1]) > seq_len // _REPETITION_DIVISOR
    )


def _select(streams: Iterable[Sequence[int]], *, n_samples: int, seq_len: int) -> list[list[int]]:
    """Keep documents already long enough, truncated to length.

    The reference's default rule, and it is a selection rather than a neutral
    one: at two thousand tokens most of a mixed corpus is discarded and what
    remains is whatever kind of text happens to be long. Matching it is a
    deliberate choice about comparability, recorded in MEMORY.md Q-003, not an
    opinion about which corpus is better.
    """
    samples: list[list[int]] = []
    for ids in streams:
        if len(ids) < seq_len:
            continue
        sample = list(ids[:seq_len])
        if _is_degenerate(sample, seq_len):
            continue
        samples.append(sample)
        if len(samples) >= n_samples:
            break
    return samples


def _pack(
    streams: Iterable[Sequence[int]],
    *,
    n_samples: int,
    seq_len: int,
    bos: int | None = None,
    eos: int | None = None,
) -> list[list[int]]:
    """Join documents end to end and chop the stream into full-length sequences.

    Per-document sentinels are stripped on the way in and one of each is put back
    on the way out, so a packed sequence looks like a document to the model
    rather than like several with their boundaries showing.

    The room for those two tokens is reserved per sequence. The reference
    reserves it in a counter that is never reset, so after enough documents the
    reservation exceeds the sequence length, the arithmetic goes negative, and
    every emitted sequence is the wrong length. See MEMORY.md Q-003.

    Consumes ``streams`` lazily and stops as soon as the budget is met, which is
    what lets the caller pass a generator that tokenizes on demand rather than
    encoding a corpus it will mostly discard.
    """
    body = seq_len - (bos is not None) - (eos is not None)
    if body < 1:
        msg = f"seq_len {seq_len} leaves no room for content once sentinels are reserved"
        raise CalibrationError(msg)

    samples: list[list[int]] = []
    buffer: list[int] = []
    for document in streams:
        ids = list(document)
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        if eos is not None and ids and ids[-1] == eos:
            ids = ids[:-1]
        buffer.extend(ids)

        while len(buffer) >= body:
            sample = buffer[:body]
            del buffer[:body]
            if bos is not None:
                sample = [bos, *sample]
            if eos is not None:
                sample = [*sample, eos]
            if not _is_degenerate(sample, seq_len):
                samples.append(sample)
            if len(samples) >= n_samples:
                return samples
    return samples


def content_hash(
    samples: Sequence[Sequence[int]],
    *,
    source: str,
    packing: Packing,
    seq_len: int,
    seed: int,
) -> str:
    """Hash the tokens and the settings that produced them.

    The tokens are hashed rather than the source text, so two models with
    different tokenizers reading the same corpus get different hashes. They are
    genuinely different calibration sets, and treating them as one is the exact
    confusion this function exists to prevent.

    Args:
        samples: The built sequences.
        source: Dataset identifier.
        packing: How documents became sequences.
        seq_len: Tokens per sequence.
        seed: Seed that fixed the document order.

    Returns:
        Sixteen hexadecimal characters. Short enough to quote in a log line, and
        far past the point where an accidental collision is a real concern.
    """
    digest = hashlib.sha256()
    header = f"{_HASH_VERSION}|{source}|{packing}|{len(samples)}|{seq_len}|{seed}\n"
    digest.update(header.encode("utf-8"))
    for sample in samples:
        # Little-endian and explicit, so the hash does not depend on the machine
        # that computed it.
        digest.update(struct.pack(f"<{len(sample)}I", *sample))
    return digest.hexdigest()[:16]


def _iter_documents(source: str, *, seed: int) -> Iterator[str]:
    """Yield the corpus one document at a time, in seeded random order.

    A local path is read directly and yielded whole, which is what pins a run to
    an exact file rather than to whatever a Hub dataset contains today. The seed
    has no effect in that case, since there is one document and no order to fix.

    Raises:
        CalibrationError: If the identifier is neither known nor a readable file,
            or if the dataset stack is not installed.
    """
    path = Path(source)
    if path.is_file():
        yield path.read_text(encoding="utf-8")
        return

    if source not in KNOWN_CORPORA:
        known = ", ".join(sorted(KNOWN_CORPORA))
        msg = f"{source!r} is neither a readable file nor one of {{{known}}}"
        raise CalibrationError(msg)

    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        msg = (
            f"building a corpus from {source!r} needs the evaluation stack. "
            f"Install it with:\n    pip install -e '.[eval]'\n"
            f"Or pass a path to a local text file instead."
        )
        raise CalibrationError(msg) from exc

    name, subset, split, column = KNOWN_CORPORA[source]
    try:
        rows = load_dataset(name, subset, split=split)
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        msg = (
            f"could not fetch {name}[{split}]: {first_line}\n"
            f"  Dataset ids move and the `datasets` library has tightened what it "
            f"accepts more\n  than once. A local text file avoids the question and "
            f"pins the corpus to an exact\n  file, which a published number needs "
            f"anyway."
        )
        raise CalibrationError(msg) from exc

    if column not in rows.column_names:
        msg = (
            f"{name}[{split}] has no {column!r} column; it has "
            f"{sorted(rows.column_names)}. The dataset changed shape, and "
            f"KNOWN_CORPORA needs updating rather than working around."
        )
        raise CalibrationError(msg)

    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    for index in order:
        text = rows[index][column]
        if text:
            yield text


def _truncating_encoder(tokenizer: object, seq_len: int | None) -> Callable[[str], Sequence[int]]:
    """An encode function that stops at ``seq_len`` tokens where it can.

    Encoding a whole document to keep its first two thousand tokens is wasted
    work on a corpus of ten thousand, and on any tokenizer with a configured
    maximum it also emits a length warning per document. That warning is a false
    alarm here and looks exactly like a real problem.

    That reasoning holds for the filtering rule only. Packing joins whole
    documents end to end, so it needs every token of each one; passing
    ``None`` returns the plain encoder for it, and a Hugging Face tokenizer's
    own configured maximum is lifted for the call so a long document is not
    cut by the tokenizer either.

    The capability is detected rather than assumed, so a plain callable with no
    options still works. Detection is by signature rather than by catching
    ``TypeError``, which would also swallow a genuine one raised inside.
    """
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        msg = f"{type(tokenizer).__name__} has no callable encode method"
        raise CalibrationError(msg)

    plain: Callable[[str], Sequence[int]] = encode
    if seq_len is None:
        return plain
    try:
        parameters = inspect.signature(encode).parameters.values()
    except (TypeError, ValueError):
        # A C-implemented callable with no introspectable signature. Rare, and
        # the conservative branch costs only some wasted tokenization.
        return plain

    accepts = any(
        parameter.name == "truncation" or parameter.kind is parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if not accepts:
        return plain
    return lambda text: encode(text, truncation=True, max_length=seq_len)


def build_calibration_set(
    tokenizer: Any,
    *,
    source: str = DEFAULT_CORPUS,
    packing: Packing = Packing.FILTER,
    n_samples: int = 128,
    seq_len: int = 2048,
    batch_size: int = 8,
    seed: int = DEFAULT_SEED,
) -> CalibrationSet:
    """Tokenize and batch a calibration corpus.

    Sampling is seeded and the seed is part of the content hash, so the same
    arguments always produce the same corpus.

    Args:
        tokenizer: The model's tokenizer.
        source: Dataset identifier, or a path to a local text file.
        packing: How documents become sequences. The default drops documents
            shorter than ``seq_len``, matching the reference implementation.
        n_samples: Sequences to draw.
        seq_len: Tokens per sequence. Nothing is padded, so every sequence is
            exactly this long.
        batch_size: Sequences per batch.
        seed: Sampling seed.

    Returns:
        The tokenized corpus.

    Raises:
        CalibrationError: If the corpus yields too little usable text for
            ``n_samples`` sequences of ``seq_len`` tokens, or if the arguments do
            not describe a corpus at all.
    """
    if n_samples < 1 or seq_len < 1 or batch_size < 1:
        msg = (
            f"n_samples, seq_len, and batch_size must all be positive; got "
            f"{n_samples}, {seq_len}, {batch_size}"
        )
        raise CalibrationError(msg)

    packing = Packing(packing)
    # Filtering keeps a document's first seq_len tokens and can stop encoding
    # there; packing uses every token of every document and must not. With
    # the truncating encoder applied to both, a packed corpus was made of
    # documents each cut to seq_len, and the remedy the error below
    # recommends for a small corpus made the corpus smaller still.
    encode = _truncating_encoder(tokenizer, None if packing is Packing.CONCAT else seq_len)
    documents = (encode(text) for text in _iter_documents(source, seed=seed))

    if packing is Packing.CONCAT:
        samples = _pack(
            documents,
            n_samples=n_samples,
            seq_len=seq_len,
            bos=getattr(tokenizer, "bos_token_id", None),
            eos=getattr(tokenizer, "eos_token_id", None),
        )
    else:
        samples = _select(documents, n_samples=n_samples, seq_len=seq_len)

    if len(samples) < n_samples:
        hint = (
            ""
            if packing is Packing.CONCAT
            else (
                "\n  The default keeps only documents already at least seq_len "
                "tokens long, which\n  discards most of a mixed corpus. "
                "packing=Packing.CONCAT uses the short ones too."
            )
        )
        msg = (
            f"{source!r} yielded {len(samples)} sequences of {seq_len} tokens, "
            f"not {n_samples}. Ask for fewer samples, a shorter sequence, or a "
            f"larger corpus; MRound will not pad the difference, because padded "
            f"tokens carry no calibration signal and still cost a forward pass "
            f"through every block.{hint}"
        )
        raise CalibrationError(msg)

    import mlx.core as mx  # noqa: PLC0415

    digest = content_hash(samples, source=source, packing=packing, seq_len=seq_len, seed=seed)
    batches = [
        mx.array(samples[start : start + batch_size], dtype=mx.int32)
        for start in range(0, len(samples), batch_size)
    ]
    return CalibrationSet(
        batches=batches,
        attention_masks=None,
        source=source,
        packing=packing,
        n_samples=len(samples),
        seq_len=seq_len,
        seed=seed,
        content_hash=digest,
    )


def save_calibration_set(calibration: CalibrationSet, path: str | Path) -> Path:
    """Write a corpus so a later run can measure against the same tokens.

    Args:
        calibration: The corpus to save.
        path: Destination file. A ``.safetensors`` suffix is added if absent,
            since that is what the format is and MLX dispatches on it.

    Returns:
        The path written.

    Raises:
        CalibrationError: If the corpus holds no batches.
    """
    import mlx.core as mx  # noqa: PLC0415

    if not calibration.batches:
        msg = "refusing to save an empty calibration set"
        raise CalibrationError(msg)

    destination = Path(path)
    if destination.suffix != ".safetensors":
        destination = destination.with_suffix(".safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)

    tokens = mx.concatenate(calibration.batches, axis=0)
    mx.save_safetensors(
        str(destination),
        {"tokens": tokens},
        metadata={
            "hash_version": _HASH_VERSION,
            "source": calibration.source,
            "packing": str(calibration.packing),
            "n_samples": str(calibration.n_samples),
            "seq_len": str(calibration.seq_len),
            "seed": str(calibration.seed),
            "batch_size": str(calibration.batch_size),
            "content_hash": calibration.content_hash,
        },
    )
    return destination


def load_calibration_set(path: str | Path) -> CalibrationSet:
    """Load a previously saved corpus, verifying its hash.

    Reusing a saved corpus rather than rebuilding it is what keeps results
    comparable across runs and across machines. The hash is recomputed from the
    tokens rather than trusted, because a file that has been edited or truncated
    is exactly the case the hash exists to catch, and it is silent otherwise.

    Args:
        path: File written by a prior :func:`save_calibration_set` run.

    Returns:
        The corpus, with its original provenance.

    Raises:
        CalibrationError: If the file is unreadable, malformed, or its hash does
            not match its contents.
    """
    import mlx.core as mx  # noqa: PLC0415

    source_path = Path(path)
    try:
        arrays, metadata = mx.load(str(source_path), return_metadata=True)
    except Exception as exc:
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        msg = f"could not read {source_path}: {first_line}"
        raise CalibrationError(msg) from exc

    if "tokens" not in arrays:
        msg = f"{source_path} has no 'tokens' array; it has {sorted(arrays)}"
        raise CalibrationError(msg)

    stored_version = metadata.get("hash_version")
    if stored_version != _HASH_VERSION:
        msg = (
            f"{source_path} was written by {stored_version!r} and this is "
            f"{_HASH_VERSION!r}. The sequence building changed, so the tokens in "
            f"that file are not the ones these arguments would produce today. "
            f"Rebuild it."
        )
        raise CalibrationError(msg)

    tokens = arrays["tokens"]
    samples = tokens.tolist()
    seq_len = int(tokens.shape[-1])
    seed = int(metadata.get("seed", "0"))
    corpus_source = metadata.get("source", str(source_path))
    packing = Packing(metadata.get("packing", Packing.FILTER))
    digest = content_hash(
        samples, source=corpus_source, packing=packing, seq_len=seq_len, seed=seed
    )
    if digest != metadata.get("content_hash"):
        msg = (
            f"{source_path} does not hash to what it claims: computed {digest}, "
            f"file says {metadata.get('content_hash')!r}. The file has been "
            f"modified or truncated, and any number measured against it would be "
            f"attributed to the wrong corpus."
        )
        raise CalibrationError(msg)

    batch_size = int(metadata.get("batch_size", str(len(samples))))
    batches = [
        tokens[start : start + batch_size] for start in range(0, tokens.shape[0], batch_size)
    ]
    return CalibrationSet(
        batches=batches,
        attention_masks=None,
        source=corpus_source,
        packing=packing,
        n_samples=int(tokens.shape[0]),
        seq_len=seq_len,
        seed=seed,
        content_hash=digest,
    )
