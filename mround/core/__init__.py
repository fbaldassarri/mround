# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Framework-agnostic quantization math.

This layer knows nothing about transformers, or about models at all. It
quantizes a weight matrix given inputs and target outputs, which is what makes
it unit-testable on synthetic data with no model loaded.
"""

from __future__ import annotations
