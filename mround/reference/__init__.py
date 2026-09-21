# Copyright 2026 The MRound Authors
# SPDX-License-Identifier: Apache-2.0
"""The framework-free reference implementation.

This package implements the complete quantization method in NumPy. It exists for
three reasons, in order of importance:

1. **It is the executable specification.** DOCUMENTATION.md section 5 states the
   formulas in prose and mathematics; this package states them in code that
   runs. Where the two disagree, the disagreement is a defect in one of them and
   gets resolved rather than tolerated.

2. **It is the oracle for the MLX implementation.** Comparing MLX against this,
   under identical inputs, isolates framework porting errors from mathematical
   errors. That separation is worth a great deal: a discrepancy against this
   reference means the port is wrong, whereas a discrepancy against Intel's
   implementation could mean either project is wrong.

3. **It runs anywhere.** No MLX, no Metal, no Apple hardware, no model. That is
   what lets the numerical core be tested in continuous integration on any
   machine, and it is why this package must never import MLX. A unit test
   enforces that.

Correctness here is prioritized absolutely over speed. This code is not intended
to quantize a real model in reasonable time; it is intended to be obviously
right. Where a clear formulation and a fast one conflict, the clear one wins.

Gradients are derived analytically and verified against finite differences in
the test suite. The finite-difference tests are not decoration: hand-derived
gradients through a straight-through estimator and a clamp are exactly the kind
of thing that is subtly wrong in a way nothing else catches.
"""

from __future__ import annotations
