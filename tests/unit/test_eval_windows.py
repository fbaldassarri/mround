# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The perplexity window schedule, which is arithmetic and needs no model.

Worth testing on its own because it is the part of a perplexity implementation
that is quietly wrong most often. Scoring a token twice lowers the result, and
nothing about the output looks wrong when it happens: the number is simply
better than it should be, which is the direction nobody investigates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mround.eval.perplexity import (
    KNOWN_DATASETS,
    PerplexityResult,
    _encode_prefix,
    _windows,
    load_evaluation_text,
)
from mround.exceptions import CalibrationError

if TYPE_CHECKING:
    from pathlib import Path

# (n_tokens, seq_len, stride)
SCHEDULES = [
    (10, 4, 4),
    (10, 4, 2),
    (100, 8, 8),
    (100, 8, 4),
    (100, 8, 1),
    (5, 8, 8),
    (2, 4, 4),
    (2049, 2048, 2048),
    (3000, 2048, 512),
    (3000, 2048, 2048),
]
OVERLAPPING = [(n, seq_len, stride) for n, seq_len, stride in SCHEDULES if stride < seq_len]


def scored_targets(n_tokens: int, seq_len: int, stride: int) -> list[int]:
    """Which target positions each window claims, flattened in order."""
    claimed: list[int] = []
    for start, n_scored in _windows(n_tokens, seq_len, stride):
        end = min(start + seq_len, n_tokens)
        claimed.extend(range(end - n_scored, end))
    return claimed


class TestWindowSchedule:
    @pytest.mark.parametrize(("n_tokens", "seq_len", "stride"), SCHEDULES)
    def test_no_target_is_scored_twice(self, n_tokens: int, seq_len: int, stride: int) -> None:
        # The property that matters. Double counting is invisible in the output
        # and always biases the result downward.
        claimed = scored_targets(n_tokens, seq_len, stride)
        assert len(set(claimed)) == len(claimed)

    @pytest.mark.parametrize(("n_tokens", "seq_len", "stride"), SCHEDULES)
    def test_no_target_is_invented(self, n_tokens: int, seq_len: int, stride: int) -> None:
        # Position zero is never a target, it is the first input, and nothing
        # beyond the last token exists.
        claimed = scored_targets(n_tokens, seq_len, stride)
        assert set(claimed) <= set(range(1, n_tokens))

    @pytest.mark.parametrize(("n_tokens", "seq_len", "stride"), SCHEDULES)
    def test_windows_never_claim_more_than_they_compute(
        self, n_tokens: int, seq_len: int, stride: int
    ) -> None:
        # A window of length L produces L-1 losses. Claiming more would slice
        # past the start of the array, which in MLX wraps rather than raising.
        for start, n_scored in _windows(n_tokens, seq_len, stride):
            available = min(start + seq_len, n_tokens) - start - 1
            assert 0 < n_scored <= available

    @pytest.mark.parametrize(("n_tokens", "seq_len", "stride"), OVERLAPPING)
    def test_overlapping_windows_score_everything(
        self, n_tokens: int, seq_len: int, stride: int
    ) -> None:
        # Only the overlapping schedules: non-overlapping windows cannot score
        # their boundary token, which the next test pins on its own, and a
        # parametrization that skipped six of ten cases every run read as six
        # environment problems rather than as the property it is.
        assert scored_targets(n_tokens, seq_len, stride) == list(range(1, n_tokens))

    def test_non_overlapping_windows_skip_exactly_the_boundary_tokens(self) -> None:
        # Not a defect, and worth pinning so nobody "fixes" it into double
        # counting. With independent windows the first token of each has no
        # predecessor inside its own window, so it cannot be predicted at all.
        n_tokens, seq_len = 100, 8
        missed = set(range(1, n_tokens)) - set(scored_targets(n_tokens, seq_len, seq_len))
        assert missed == set(range(seq_len, n_tokens, seq_len))

    def test_a_shorter_stride_scores_at_least_as_many_tokens(self) -> None:
        counts = [len(scored_targets(1000, 128, stride)) for stride in (128, 64, 32, 16)]
        assert counts == sorted(counts)

    def test_terminates_on_input_too_short_to_score(self) -> None:
        assert list(_windows(1, 8, 8)) == []
        assert list(_windows(0, 8, 8)) == []


class TestResultReporting:
    def test_describe_carries_the_settings(self) -> None:
        # A perplexity without its settings is not a number anyone can check,
        # so the settings travel with it rather than living in a log line.
        result = PerplexityResult(
            perplexity=12.5, n_tokens=2048, dataset="wikitext2", seq_len=2048, stride=512
        )
        described = result.describe()
        for fragment in ("12.5", "2048", "wikitext2", "512"):
            assert fragment in described


class TestDatasetResolution:
    def test_a_local_file_is_read_directly(self, tmp_path: Path) -> None:
        # What makes an evaluation pinnable to an exact file rather than to
        # whatever a Hub dataset contains today.
        target = tmp_path / "held_out.txt"
        target.write_text("the quick brown fox", encoding="utf-8")
        assert load_evaluation_text(str(target)) == "the quick brown fox"

    def test_an_unknown_identifier_says_what_is_known(self) -> None:
        with pytest.raises(CalibrationError, match="wikitext2"):
            load_evaluation_text("not-a-dataset-and-not-a-file")

    def test_the_known_datasets_are_fully_specified(self) -> None:
        # Each entry must name the dataset, the configuration, and the split.
        # A missing split silently scores the training set, which is the one
        # mistake in this file that would invalidate every published number.
        for name, spec in KNOWN_DATASETS.items():
            assert len(spec) == 3, name
            assert spec[2] in {"test", "validation"}, f"{name} scores {spec[2]!r}"

    def test_every_repository_id_is_namespaced(self) -> None:
        # `datasets` version 5 rejects a bare name outright, and before that an
        # unqualified id was a redirect that could expire. This caught nothing
        # when it was written because the ids had just been fixed; it exists so
        # the next one is caught before it reaches a machine.
        for name, (repo, _, _) in KNOWN_DATASETS.items():
            assert "/" in repo, f"{name} points at {repo!r}, which has no namespace"
            assert not repo.startswith("/"), repo
            assert not repo.endswith("/"), repo


class FakeTokenizer:
    """A tokenizer with a configurable characters-per-token ratio."""

    def __init__(self, chars_per_token: int) -> None:
        self.ratio = chars_per_token

    def encode(self, text: str) -> list[int]:
        return list(range(len(text) // self.ratio))


class TestPrefixEncoding:
    """Tokenizing only as much text as the token budget needs.

    An optimization that truncates the evaluation set silently would lower
    perplexity and look like an improvement, so the property under test is that
    the shortcut returns exactly what encoding everything and slicing would.
    """

    @pytest.mark.parametrize("chars_per_token", [1, 2, 4, 8, 16, 32])
    @pytest.mark.parametrize("needed", [10, 2049, 8193])
    def test_matches_encoding_everything_then_slicing(
        self, chars_per_token: int, needed: int
    ) -> None:
        tokenizer = FakeTokenizer(chars_per_token)
        text = "x" * 100_000
        assert _encode_prefix(tokenizer, text, needed) == tokenizer.encode(text)[:needed]

    def test_a_dense_tokenizer_is_not_truncated(self) -> None:
        # One token per character defeats any fixed bytes-per-token guess, which
        # is why the slice grows rather than being estimated. Byte-level
        # tokenizers on non-Latin scripts land here.
        tokenizer = FakeTokenizer(1)
        assert len(_encode_prefix(tokenizer, "y" * 50_000, 40_000)) == 40_000

    def test_no_budget_encodes_everything(self) -> None:
        tokenizer = FakeTokenizer(4)
        assert len(_encode_prefix(tokenizer, "z" * 4_000, None)) == 1_000

    def test_a_budget_larger_than_the_text_returns_what_there_is(self) -> None:
        tokenizer = FakeTokenizer(4)
        assert len(_encode_prefix(tokenizer, "z" * 400, 10_000)) == 100
