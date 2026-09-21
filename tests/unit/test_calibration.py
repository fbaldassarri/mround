# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Corpus building and the content hash that identifies it.

All of it is arithmetic over lists of integers, so all of it is testable without
MLX, without a tokenizer, and without a download. That matters more here than it
looks: the calibration set is an input to every number this project publishes,
and the ways it can be quietly wrong are sequence building that loses or repeats
tokens, and a hash that fails to distinguish two corpora that differ.

A hash that collides is the worse of the two. It would let a run measured on one
corpus be compared against a run measured on another, and the comparison would
look valid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from mround.exceptions import CalibrationError
from mround.pipeline.calibration import (
    DEFAULT_CORPUS,
    DEFAULT_SEED,
    KNOWN_CORPORA,
    Packing,
    _is_degenerate,
    _iter_documents,
    _pack,
    _select,
    _truncating_encoder,
    build_calibration_set,
    content_hash,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# Pins the hash's wire format. If sequence building or the header ever changes,
# this fails and _HASH_VERSION has to be bumped in the same commit, which is the
# point: a silently changed hash lets an old saved corpus pass for a new one.
GOLDEN = "ad445779857ff5e8"


class CountingDocuments:
    """A document source that records how much of itself was consumed."""

    def __init__(self, documents: list[list[int]]) -> None:
        self.documents = documents
        self.pulled = 0

    def __iter__(self) -> Iterator[list[int]]:
        """Yield documents, counting each one as it leaves."""
        for document in self.documents:
            self.pulled += 1
            yield document


class TestSelection:
    """The default rule, which is the reference implementation's."""

    def test_documents_are_truncated_to_length(self) -> None:
        assert _select([list(range(10))], n_samples=1, seq_len=4) == [[0, 1, 2, 3]]

    def test_short_documents_are_dropped_not_padded(self) -> None:
        # The property that makes an attention mask unnecessary, and the one
        # that makes this rule a selection: at two thousand tokens most of a
        # mixed corpus fails it.
        assert _select([[1, 2, 3]], n_samples=1, seq_len=4) == []

    def test_the_budget_stops_consumption(self) -> None:
        source = CountingDocuments([list(range(8)) for _ in range(1000)])
        assert len(_select(source, n_samples=3, seq_len=4)) == 3
        assert source.pulled == 3

    def test_a_document_of_exactly_the_right_length_is_kept(self) -> None:
        # The boundary the reference's comparison sits on, and the one an
        # off-by-one would move.
        assert _select([[1, 2, 3, 4]], n_samples=1, seq_len=4) == [[1, 2, 3, 4]]


class TestPacking:
    """The opt-in rule, which uses the text the default discards."""

    def test_documents_are_concatenated_then_chopped(self) -> None:
        packed = _pack([[1, 2, 3], [4, 5, 6, 7, 8]], n_samples=2, seq_len=4)
        assert packed == [[1, 2, 3, 4], [5, 6, 7, 8]]

    def test_every_sequence_is_exactly_full_length(self) -> None:
        packed = _pack([list(range(37))] * 5, n_samples=6, seq_len=8)
        assert all(len(sample) == 8 for sample in packed)

    def test_no_token_is_used_twice(self) -> None:
        packed = _pack([list(range(100))], n_samples=5, seq_len=10)
        flat = [token for sample in packed for token in sample]
        assert flat == list(range(50))

    def test_the_tail_is_dropped_rather_than_padded(self) -> None:
        assert _pack([[1, 2, 3]], n_samples=1, seq_len=4) == []

    def test_a_short_corpus_returns_what_it_has(self) -> None:
        # The caller raises on this; packing itself reports rather than invents,
        # so the error message can say how many sequences there actually were.
        assert len(_pack([list(range(20))], n_samples=10, seq_len=4)) == 5

    def test_consumption_stops_once_the_budget_is_met(self) -> None:
        # The reason this takes an iterable. Tokenizing ten thousand documents
        # to keep four is minutes of wall clock for nothing.
        source = CountingDocuments([list(range(4)) for _ in range(1000)])
        assert len(_pack(source, n_samples=2, seq_len=4)) == 2
        assert source.pulled == 2

    def test_one_long_document_can_fill_the_whole_corpus(self) -> None:
        source = CountingDocuments([list(range(1000)), list(range(1000))])
        assert len(_pack(source, n_samples=4, seq_len=100)) == 4
        assert source.pulled == 1


class TestSentinelReservation:
    """Room for the sentinels is reserved per sequence, not accumulated.

    The reference accumulates it in a counter that is never reset. After enough
    documents the reservation exceeds the sequence length, the slice arithmetic
    goes negative, and every sequence it emits is the wrong length, which the
    length filter then rejects, which empties the corpus. It is worth a test
    precisely because it only appears at scale.
    """

    def test_sentinels_are_attached_and_the_length_still_holds(self) -> None:
        packed = _pack([list(range(1, 21))], n_samples=2, seq_len=6, bos=101, eos=102)
        assert packed == [[101, 1, 2, 3, 4, 102], [101, 5, 6, 7, 8, 102]]

    def test_a_document_carrying_its_own_sentinels_is_not_doubled(self) -> None:
        document = [101, 1, 2, 3, 4, 5, 6, 7, 8, 102]
        packed = _pack([document], n_samples=2, seq_len=6, bos=101, eos=102)
        assert packed == [[101, 1, 2, 3, 4, 102], [101, 5, 6, 7, 8, 102]]

    def test_the_reservation_does_not_grow_with_the_corpus(self) -> None:
        # The failing case in the reference, reproduced at a scale that would
        # trigger it: many documents, each carrying both sentinels.
        documents = [[101, *range(1, 9), 102] for _ in range(500)]
        packed = _pack(documents, n_samples=100, seq_len=6, bos=101, eos=102)
        assert len(packed) == 100
        assert all(len(sample) == 6 for sample in packed)
        assert all(sample[0] == 101 and sample[-1] == 102 for sample in packed)

    def test_a_sequence_too_short_to_hold_its_sentinels_refuses(self) -> None:
        with pytest.raises(CalibrationError, match="no room"):
            _pack([[1, 2, 3]], n_samples=1, seq_len=2, bos=101, eos=102)


class TestDegenerateFilter:
    def test_a_repeated_token_over_half_the_sequence_is_rejected(self) -> None:
        assert _is_degenerate([9, 9, 9, 1, 2, 9], seq_len=6)

    def test_an_ordinary_sequence_is_kept(self) -> None:
        assert not _is_degenerate([1, 2, 3, 4, 5, 6], seq_len=6)

    def test_exactly_half_is_kept(self) -> None:
        # The reference's comparison is strictly greater than, and moving it
        # would discard ordinary text: half a sequence of one token is common
        # in code and in tabular data. Three nines in six positions is exactly
        # half, so this must survive.
        assert not _is_degenerate([9, 1, 9, 2, 3, 9], seq_len=6)

    def test_only_the_final_token_is_counted(self) -> None:
        # A sequence dominated by a token that is not the last one passes. That
        # is the reference's rule, and the rule is aimed at trailing runs of
        # padding or whitespace rather than at repetition in general.
        assert not _is_degenerate([9, 9, 9, 9, 9, 1], seq_len=6)

    def test_very_short_sequences_are_exempt(self) -> None:
        # Below three tokens almost anything has some token in half its
        # positions, so the rule would reject text it was never aimed at.
        assert not _is_degenerate([7, 7], seq_len=2)

    def test_the_filter_applies_to_selection(self) -> None:
        assert _select([[9, 9, 9, 9]], n_samples=1, seq_len=4) == []

    def test_the_filter_applies_to_packing(self) -> None:
        assert _pack([[9, 9, 9, 9]], n_samples=1, seq_len=4) == []


class TestContentHash:
    def test_the_same_tokens_hash_the_same_way(self) -> None:
        args: dict[str, Any] = {
            "source": "a",
            "packing": Packing.FILTER,
            "seq_len": 2,
            "seed": 0,
        }
        assert content_hash([[1, 2], [3, 4]], **args) == content_hash([[1, 2], [3, 4]], **args)

    @pytest.mark.parametrize(
        ("samples", "source", "packing", "seq_len", "seed"),
        [
            ([[1, 2], [3, 5]], "a", Packing.FILTER, 2, 0),
            ([[1, 2]], "a", Packing.FILTER, 2, 0),
            ([[1, 2], [3, 4]], "b", Packing.FILTER, 2, 0),
            ([[1, 2], [3, 4]], "a", Packing.CONCAT, 2, 0),
            ([[1, 2], [3, 4]], "a", Packing.FILTER, 2, 1),
        ],
    )
    def test_anything_that_changes_the_corpus_changes_the_hash(
        self,
        samples: list[list[int]],
        source: str,
        packing: Packing,
        seq_len: int,
        seed: int,
    ) -> None:
        baseline = content_hash(
            [[1, 2], [3, 4]], source="a", packing=Packing.FILTER, seq_len=2, seed=0
        )
        computed = content_hash(samples, source=source, packing=packing, seq_len=seq_len, seed=seed)
        assert computed != baseline

    def test_the_packing_mode_is_part_of_the_identity(self) -> None:
        # Two corpora built from the same documents by different rules are
        # different calibration sets, and the same tokens can come out of both.
        tokens = [[1, 2, 3, 4]]
        assert content_hash(
            tokens, source="a", packing=Packing.FILTER, seq_len=4, seed=0
        ) != content_hash(tokens, source="a", packing=Packing.CONCAT, seq_len=4, seed=0)

    def test_token_order_matters(self) -> None:
        # A hash over a set or a sum would miss this, and two corpora with the
        # same tokens in a different order are different calibration sets.
        assert content_hash(
            [[1, 2]], source="a", packing=Packing.FILTER, seq_len=2, seed=0
        ) != content_hash([[2, 1]], source="a", packing=Packing.FILTER, seq_len=2, seed=0)

    def test_sample_boundaries_matter(self) -> None:
        # Four tokens as one sequence and as two are different corpora, and a
        # hash of the flattened stream would call them equal.
        assert content_hash(
            [[1, 2, 3, 4]], source="a", packing=Packing.FILTER, seq_len=4, seed=0
        ) != content_hash([[1, 2], [3, 4]], source="a", packing=Packing.FILTER, seq_len=2, seed=0)

    def test_the_format_is_pinned(self) -> None:
        assert (
            content_hash(
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                source="unit-test",
                packing=Packing.FILTER,
                seq_len=4,
                seed=DEFAULT_SEED,
            )
            == GOLDEN
        )

    def test_it_is_short_enough_to_quote(self) -> None:
        digest = content_hash([[1]], source="a", packing=Packing.FILTER, seq_len=1, seed=0)
        assert len(digest) == 16
        assert all(character in "0123456789abcdef" for character in digest)


class PlainTokenizer:
    """A tokenizer whose encode takes nothing but the text."""

    def encode(self, text: str) -> list[int]:
        """One token per character."""
        return [ord(character) for character in text]


class TruncatingTokenizer:
    """A tokenizer that accepts the options a Hugging Face one does."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        """One token per character, honoring a maximum length."""
        ids = [ord(character) for character in text]
        limit = kwargs.get("max_length")
        return ids[:limit] if kwargs.get("truncation") and limit else ids


class TestEncoderDetection:
    def test_a_tokenizer_that_can_truncate_is_asked_to(self) -> None:
        encode = _truncating_encoder(TruncatingTokenizer(), 4)
        assert list(encode("abcdefgh")) == [ord("a"), ord("b"), ord("c"), ord("d")]

    def test_a_tokenizer_that_cannot_is_not(self) -> None:
        # Detection by signature rather than by catching TypeError, which would
        # also swallow a genuine one raised inside the tokenizer.
        encode = _truncating_encoder(PlainTokenizer(), 4)
        assert len(list(encode("abcdefgh"))) == 8

    def test_something_that_cannot_encode_at_all_says_so(self) -> None:
        with pytest.raises(CalibrationError, match="encode"):
            _truncating_encoder(object(), 4)

    def test_no_length_means_the_plain_encoder(self) -> None:
        encode = _truncating_encoder(TruncatingTokenizer(), None)
        assert len(list(encode("abcdefgh"))) == 8

    @pytest.mark.needs_mlx
    def test_packing_uses_whole_documents_with_a_truncating_tokenizer(self, tmp_path: Path) -> None:
        # The packed corpus is the remedy the filtering error recommends for a
        # small corpus, and with a Hugging Face tokenizer (which truncates on
        # request) it used to cut every document to seq_len before joining
        # them, so one 30 token document packed to a single 4 token sequence.
        #
        # The only test in this file that needs MLX: the other four in this
        # class stop at the encoder, while this one runs the whole builder,
        # which materializes the corpus as MLX arrays. Its neighbours are
        # framework free and must stay collectable without MLX, so the gate is
        # on the test rather than on the module.
        pytest.importorskip("mlx.core", reason="build_calibration_set produces MLX arrays")

        target = tmp_path / "corpus.txt"
        target.write_text("the quick brown fox jumps over", encoding="utf-8")
        packed = build_calibration_set(
            TruncatingTokenizer(),
            source=str(target),
            n_samples=5,
            seq_len=4,
            batch_size=5,
            packing=Packing.CONCAT,
        )
        assert packed.n_samples == 5
        assert packed.seq_len == 4


class TestCorpusResolution:
    def test_a_local_file_is_one_document(self, tmp_path: Path) -> None:
        # What pins a run to an exact file rather than to whatever a Hub dataset
        # contains today.
        target = tmp_path / "corpus.txt"
        target.write_text("the quick brown fox", encoding="utf-8")
        assert list(_iter_documents(str(target), seed=0)) == ["the quick brown fox"]

    def test_an_unknown_identifier_says_what_is_known(self) -> None:
        with pytest.raises(CalibrationError, match=DEFAULT_CORPUS):
            next(_iter_documents("not-a-corpus-and-not-a-file", seed=0))

    def test_the_default_corpus_is_one_of_the_known_ones(self) -> None:
        assert DEFAULT_CORPUS in KNOWN_CORPORA

    def test_every_corpus_is_fully_specified(self) -> None:
        # Repository, configuration, split, and text column. A wrong column name
        # fails late, and a bare repository id is rejected outright by version 5
        # of the datasets library.
        for name, spec in KNOWN_CORPORA.items():
            repo, _, split, column = spec
            assert "/" in repo, f"{name} points at {repo!r}, which has no namespace"
            assert split, f"{name} names no split"
            assert column, f"{name} names no text column"
