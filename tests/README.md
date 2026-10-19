# Tests

Phase 0 infrastructure is checked with `py -3.11 scripts/verify_phase0.py`
(Linux: `python3 scripts/verify_phase0.py`). Meaningful baseline, OOF leakage,
preprocessing, frozen-gradient, serialization, and integration tests will be
added alongside the code they verify; see [the plan](../docs/PROJECT_PLAN.md).
