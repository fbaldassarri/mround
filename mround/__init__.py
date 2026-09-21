# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""MRound: weight-only quantization for large language models on Apple Silicon.

MRound is an independent implementation of the SignRound v2 method. It quantizes
transformer weights to 8, 4, and 2 bits by learning per-weight rounding
perturbations and per-group clipping coefficients with signed gradient descent,
so that each transformer block reproduces its original output as closely as
possible.

The public API is :mod:`mround.api`. Everything else is internal and may change
without notice.

See DOCUMENTATION.md for the specification and ROADMAP.md for what is built.
"""

from __future__ import annotations

__version__ = "0.1.0a1"

__all__ = ["__version__"]
