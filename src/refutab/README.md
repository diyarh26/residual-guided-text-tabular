# Original implementation

`baseline.py` orchestrates the original historical MulTaBench frozen baseline;
it does not implement a replacement baseline learner or ReFuTab fusion.
`data.py` declares feature roles and stable row identity; `splits.py` persists
partition assignments. `teacher.py` implements the structured-only three-fold
OOF CatBoost API and checked model persistence. `teacher_runner.py` orchestrates
audited configured runs without predicting the outer test. See
[the Phase 2 API](../../docs/PHASE2.md). Full ReFuTab fusion and its final
prediction API remain Phase 3 work.
