# Tests

Infrastructure: `py -3.11 scripts/verify_phase0.py` (Linux: `python3`).

After locked setup and `scripts/prepare_phase1.py --skip-model`, run
`uv run --frozen python -m unittest discover -s tests -v`. Phase 1 tests exercise
original split/metric behavior, training-only PCA and unknown categories, frozen
E5 inference mode, and run safety. Deterministic tiny encoder substitutes occur
only inside unit tests; the recorded baseline must use real pinned E5 weights.
No network/model download is performed by the tests themselves. OOF teacher,
fusion, serialization, and later integration tests belong to their future phases.
