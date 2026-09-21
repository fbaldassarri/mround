# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures and collection rules.

Tests are organized by what they need, not by what they cover:

- ``unit`` needs nothing. Synthetic data, no model, no MLX where avoidable.
  These run everywhere, including on machines that are not Macs.
- ``parity`` needs the dumped reference corpus in ``tests/fixtures``. Skipped
  when it is absent, since generating it requires the reference stack.
- ``integration`` needs MLX and a real model. Slow.

Anything mathematical belongs in ``unit`` with a hand-computed expectation. A
test that loads a real model to verify arithmetic is in the wrong place.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PARITY_CORPUS_DIR = FIXTURES_DIR / "parity_corpus"


def _mlx_available() -> bool:
    """Whether MLX imports on this host.

    Deliberately a probe rather than a platform check: what matters is whether
    the tests can run, not what the machine claims to be. ``find_spec`` raises
    rather than returning ``None`` when the parent package is absent, which is
    exactly the case on every machine that is not a Mac.
    """
    try:
        return importlib.util.find_spec("mlx.core") is not None
    except (ImportError, ValueError):
        return False


HAVE_MLX = _mlx_available()
HAVE_PARITY_CORPUS = PARITY_CORPUS_DIR.is_dir() and any(PARITY_CORPUS_DIR.iterdir())


def pytest_collection_modifyitems(
    config: pytest.Config,  # noqa: ARG001
    items: list[pytest.Item],
) -> None:
    """Skip tests whose prerequisites are absent, rather than failing them."""
    skip_mlx = pytest.mark.skip(reason="MLX is unavailable on this host")
    skip_parity = pytest.mark.skip(
        reason=(
            "no parity corpus in tests/fixtures/parity_corpus; "
            "generate it with the reference stack (ROADMAP.md Phase 0)"
        )
    )
    for item in items:
        if "needs_mlx" in item.keywords and not HAVE_MLX:
            item.add_marker(skip_mlx)
        if "parity" in item.keywords and not HAVE_PARITY_CORPUS:
            item.add_marker(skip_parity)
        if "integration" in item.keywords and not HAVE_MLX:
            item.add_marker(skip_mlx)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Directory holding test fixtures."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def parity_corpus_dir() -> Path:
    """Directory holding the dumped reference corpus."""
    return PARITY_CORPUS_DIR


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """A scratch directory for a written checkpoint."""
    out = tmp_path / "checkpoint"
    out.mkdir()
    return out
