# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The scale initialization a string symmetry resolves to.

``QuantScheme`` and ``plan_mixed_precision`` both accept ``"sym"`` for the
symmetric range and coerce it, so ``default_scale_init`` must treat the string
exactly as the member: an identity test against the enum would send a 2 bit
``api.quantize(..., symmetry="sym")`` to the observed range, the configuration
D-030 measured at 77,365 perplexity.
"""

from __future__ import annotations

import pytest

from mround.api import default_scale_init
from mround.schemes import ScaleInit, Symmetry


@pytest.mark.parametrize("symmetry", [Symmetry.SYMMETRIC, "sym"])
def test_a_low_width_searches_whatever_spelling_the_symmetry_has(symmetry: Symmetry) -> None:
    assert default_scale_init(2, symmetry) is ScaleInit.SEARCHED


@pytest.mark.parametrize("symmetry", [Symmetry.ASYMMETRIC, "asym"])
def test_an_asymmetric_spelling_still_keeps_the_observed_range(symmetry: Symmetry) -> None:
    assert default_scale_init(2, symmetry) is ScaleInit.OBSERVED_RANGE
