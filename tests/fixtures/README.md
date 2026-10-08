# Test fixtures

Generated, not committed. Two kinds are reserved here, and nothing in the tree
writes either yet:

- `parity_corpus/`: dumped reference activations, scales, and integer codes.
  Its generator and the tests that would read it are planned, not built; see
  `tests/parity/README.md`.
- `calibration/`: frozen calibration corpora, identified by content hash so
  that two runs quoting the same hash are comparable.

Both are gitignored. No test reads either today; the parity tests, once
written, skip themselves when the corpus is absent, so a fresh clone keeps a
green suite.
