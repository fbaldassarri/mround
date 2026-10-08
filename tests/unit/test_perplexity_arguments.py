# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Arguments ``evaluate_perplexity`` must refuse before it scores anything.

A token budget below one either scored most of the set anyway (a negative
budget sliced the text from the end) or blamed the data set for the argument,
and a one token window divided by zero. Both are refused here before the model
is touched, so no model and no MLX are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mround.eval.perplexity import evaluate_perplexity
from mround.exceptions import CalibrationError

if TYPE_CHECKING:
    from pathlib import Path


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(c) % 50 for c in text]


@pytest.fixture
def text(tmp_path: Path) -> str:
    path = tmp_path / "set.txt"
    path.write_text("lorem ipsum dolor sit amet " * 40, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("max_tokens", [0, -5])
def test_a_token_budget_below_one_is_refused(text: str, max_tokens: int) -> None:
    with pytest.raises(CalibrationError, match="max_tokens"):
        evaluate_perplexity(None, _Tokenizer(), dataset=text, max_tokens=max_tokens)


def test_a_one_token_window_is_refused(text: str) -> None:
    with pytest.raises(CalibrationError, match="seq_len"):
        evaluate_perplexity(None, _Tokenizer(), dataset=text, seq_len=1)
