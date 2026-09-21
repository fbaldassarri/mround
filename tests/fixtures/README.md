# Test fixtures

Generated, not committed. Two kinds live here:

- `parity_corpus/`: dumped reference activations, scales, and integer codes.
  See `tests/parity/README.md` for how to produce it.
- `calibration/`: frozen calibration corpora, identified by content hash so
  that two runs quoting the same hash are comparable.

Both are gitignored. A fresh clone has a green test suite because the tests that
need them skip themselves.
