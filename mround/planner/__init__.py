# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Sensitivity scoring and mixed-precision allocation.

This layer knows nothing about MLX. It consumes scores and costs and produces an
allocation, which makes it testable against hand-computed examples, including
ones where greedy allocation is provably wrong.
"""

from __future__ import annotations
