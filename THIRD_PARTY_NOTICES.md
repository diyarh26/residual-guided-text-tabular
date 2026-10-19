# Attribution and redistribution boundaries

Original ReFuTab documentation and future original code are MIT licensed by
Diyar Husayyan; see [LICENSE](LICENSE). That grant does not relicense external
material, trained models, or datasets.

## MulTaBench

This research builds on **MulTaBench: Benchmarking Multimodal Tabular Learning
with Text and Image**, Arazi et al., 2026.

- Paper: https://arxiv.org/abs/2605.10616
- Historical paper: https://arxiv.org/abs/2605.10616v1
- Source: https://github.com/alanarazi7/MulTaBench
- Pinned tree: https://github.com/alanarazi7/MulTaBench/tree/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca

Only provenance metadata, original analysis, and original orchestration are
tracked here. The selected
historical tree has no LICENSE file. Current upstream has an MIT license added
later; see [UPSTREAM.md](docs/UPSTREAM.md). Preserve the upstream copyright
notice and establish how the later license applies before redistributing
historical implementation files. A later license does not change our pin.

## Course material

The eight-page Technion course instructions are local reference material.
Redistribution permission has not been established. The PDF and its rendered
or extracted intermediates are excluded from Git. Original requirements notes
are in [COURSE_REQUIREMENTS.md](docs/COURSE_REQUIREMENTS.md).

## Phase 1 dependencies and artifacts

Phase 1 executes an ignored, unmodified historical source checkout; it does not
vendor or redistribute those implementation files. The later MIT declaration
is preserved as license evidence without relabeling the historical tree. This
runtime-only integration leaves the original source clearly distinguishable.

E5-small-v2's pinned model card declares MIT; use revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52`. Model weights remain ignored. The
version-1 Product Sentiment pilot has nonstandard source license declarations;
its table is downloaded only for the user's academic evaluation and is not
redistributed here. See [dataset notices](docs/DATASETS.md).

`pyproject.toml` and `uv.lock` record the selected dependency versions and
artifact hashes. Dependencies keep their own licenses. Future CatBoost/other
learner and model-artifact terms must be reviewed when added. Do not upload
restricted data with model artifacts.
