# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Where ``save_calibration_set`` writes, for names that already contain dots.

The suffix used to be substituted rather than added, so ``pile.seq2048`` was
written as ``pile.safetensors`` and two names differing only after a dot
overwrote each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("mlx.core", reason="saving a corpus writes MLX arrays")

import mlx.core as mx

from mround.pipeline.calibration import (
    CalibrationSet,
    Packing,
    content_hash,
    load_calibration_set,
    save_calibration_set,
)

if TYPE_CHECKING:
    from pathlib import Path

SAMPLES = [[3, 1, 4, 1, 5, 9, 2, 6]]


def _corpus() -> CalibrationSet:
    digest = content_hash(SAMPLES, source="local", packing=Packing.FILTER, seq_len=8, seed=42)
    return CalibrationSet(
        batches=[mx.array(SAMPLES, dtype=mx.int32)],
        attention_masks=None,
        source="local",
        packing=Packing.FILTER,
        n_samples=1,
        seq_len=8,
        seed=42,
        content_hash=digest,
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("corpus", "corpus.safetensors"),
        ("corpus.safetensors", "corpus.safetensors"),
        ("pile.seq2048", "pile.seq2048.safetensors"),
        ("qwen-0.5b", "qwen-0.5b.safetensors"),
    ],
)
def test_the_suffix_is_added_rather_than_substituted(
    tmp_path: Path, name: str, expected: str
) -> None:
    written = save_calibration_set(_corpus(), tmp_path / name)
    assert written.name == expected
    assert load_calibration_set(written).content_hash == _corpus().content_hash


def test_two_names_differing_after_a_dot_are_two_files(tmp_path: Path) -> None:
    first = save_calibration_set(_corpus(), tmp_path / "run.seed42")
    second = save_calibration_set(_corpus(), tmp_path / "run.seed43")
    assert first != second
    assert first.is_file()
    assert second.is_file()
