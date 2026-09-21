# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the block loop's tuning recipe.

The two readings of DOCUMENTATION.md section 5.4 under a mixed plan, which is
MEMORY.md D-043. Everything here is a pure function of a configuration and a
set of widths, so no model is loaded and no block is run.
"""

from __future__ import annotations

import pytest

from mround.schemes import TuningConfig

pytest.importorskip("mlx.core", reason="the runner is the MLX layer")

from mround.pipeline.runner import block_recipe

CONFIG = TuningConfig(iters=200)
MIXED = {"attn": 2, "mlp": 3, "head": 4}


class TestBlockRecipe:
    def test_the_base_width_sets_everything_by_default(self) -> None:
        # What every published mixed measurement was produced with: a 4 bit
        # base means 1.0/iters and no outlier mask for the 2 bit layers too.
        suppress, rates = block_recipe(CONFIG, 4, MIXED, per_layer=False)
        assert suppress is False
        assert set(rates) == set(MIXED)
        assert all(rate == pytest.approx(1.0 / 200) for rate in rates.values())

    def test_per_layer_gives_each_width_its_own_rate(self) -> None:
        suppress, rates = block_recipe(CONFIG, 4, MIXED, per_layer=True)
        assert rates["attn"] == pytest.approx(2.0 / 200)
        assert rates["mlp"] == pytest.approx(2.0 / 200)
        assert rates["head"] == pytest.approx(1.0 / 200)
        # The loss is one objective over the whole block, so the outlier rule
        # cannot be per layer. It follows the narrowest width present, which is
        # what produces the errors suppression exists to mask.
        assert suppress is True

    def test_the_base_width_still_decides_a_block_with_no_tunable_layers(self) -> None:
        suppress, rates = block_recipe(CONFIG, 2, {}, per_layer=True)
        assert rates == {}
        assert suppress is True

    @pytest.mark.parametrize("base", [2, 3, 4, 8])
    def test_a_uniform_block_is_unaffected_by_the_switch(self, base: int) -> None:
        # The switch is a mixed precision question. With one width in the block
        # both readings must agree exactly, which is what makes it safe to
        # leave on for a uniform run.
        widths = {"a": base, "b": base}
        assert block_recipe(CONFIG, base, widths, per_layer=False) == block_recipe(
            CONFIG, base, widths, per_layer=True
        )

    def test_an_explicit_rate_overrides_both_readings(self) -> None:
        config = TuningConfig(iters=200, lr=0.05)
        for per_layer in (False, True):
            _, rates = block_recipe(config, 4, MIXED, per_layer=per_layer)
            assert set(rates.values()) == {0.05}
