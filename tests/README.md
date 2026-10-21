# Tests

Infrastructure: `py -3.11 scripts/verify_phase0.py` (Linux: `python3`).

After locked setup and `scripts/prepare_phase1.py --skip-model`, run
`uv run --frozen python -m unittest discover -s tests -v`. Phase 1 tests exercise
original split/metric behavior, training-only PCA and unknown categories, frozen
E5 inference mode, and run safety. Deterministic tiny encoder substitutes occur
only inside unit tests; the recorded baseline must use real pinned E5 weights.
No network/model download is performed by the tests themselves. Phase 2 adds
real tiny CatBoost checks for regression, binary, and multiclass tasks;
instrumented fitting exclusion, fixed-fold own-label perturbation, held-out
label/feature independence, strict feature/row ordering, missing/unseen values,
class/offset conventions, and checked save/load. Runner tests exercise real
training with text/target/ID exclusion, preserved failures and overwrite
protection, and exact historical split reconstruction without model evaluation.
Phase 3 checks frozen E5, PCA/structured fit scope, identity gating, all three
residual losses and the direct control, detached inputs/offsets, exact CPU
dropout/shuffle checkpoint continuation, regression/binary/multiclass pipeline
execution, test-label/feature perturbation, bundle checksums/semantic alignment,
and CSV prediction. Text substitutes are explicit in unit tests; real frozen
E5 is exercised separately by the recorded synthetic smoke. No synthetic metric
establishes benchmark improvement. GPU and nested benchmark evaluation remain
unexecuted at the Phase 3 checkpoint.

Phase 4 adds versioned-download/path/overwrite checks, original and grouped
split/OOF exclusions, metric edge cases, frozen-cache bindings/corruption,
source-compatible PCA persistence, and TAR source/patch/checkpoint tests with
explicit tiny CPU fixtures. A native PEFT weight round trip checks recovery
after training but before PCA. Runner integration exercises the all-stage test
barrier, preserved failures/new continuation attempts, artifact inventories,
and recovery of durable test predictions. Report tests block decisions with
incomplete/nonfinite/misaligned comparisons. Notebook tests exercise the SHA
and launch guards. These tests do not claim actual CUDA/Colab execution.

Controlled scaling tests cover deterministic nested class-stratified prefixes,
fixed grouped V/D and old-test exclusions, grouped OOF feasibility, held-out
label invariance, real tiny gate/teacher/PCA/downstream fitting, strict run
binding, durable attempt/assessment recovery and corruption rejection. A full
saved-artifact fixture audits 108 score sets, 12 teacher bundles and 96 fit
scopes without new inference. Frozen copies substitute for TAR only when
testing the ledger barrier; they do not validate GPU TAR training.

The stdlib private audit transfer checks exact post-transfer audit equivalence,
excluded neural checkpoints/caches, payload corruption, unsafe paths, missing
manifests and overwrite protection. The CPU transfer notebook compiles without
ML dependencies or a launch command; these tests do not perform a real transfer.
