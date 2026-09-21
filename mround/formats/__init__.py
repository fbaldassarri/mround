# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""Packing and checkpoint export.

Cross-cutting: serializes what the stack produces and depends only on the
kernels layer.
"""

from __future__ import annotations
