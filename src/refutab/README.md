# Original implementation

`baseline.py` orchestrates the original historical MulTaBench frozen baseline;
it does not implement a replacement baseline learner or ReFuTab fusion.
`data.py` declares feature roles and stable row identity; `splits.py` persists
partition assignments. `teacher.py` implements the structured-only three-fold
OOF CatBoost API and checked model persistence. `teacher_runner.py` orchestrates
audited configured runs without predicting the outer test. See
[the Phase 2 API](../../docs/PHASE2.md). Phase 3 adds `features.py`, `fusion.py`,
`downstream.py` and `pipeline.py`, with configured runs in `fusion_runner.py`.
`ReFuTabPipeline.predict` loads frozen E5, saved preprocessing, gate and
LightGBM; it needs no teacher or temporary correction head. See
[the Phase 3 API](../../docs/PHASE3.md) for training, persistence and exact
verification boundaries. Benchmark comparisons remain later work.
