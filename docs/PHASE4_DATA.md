# Phase 4 pilot data preparation

Prepared October 19, 2026 (Asia/Jerusalem), before any new pilot fitting or
comparison scores. The two-dataset choice and complete feature roles are
locked in [phase4_datasets.json](../configs/phase4_datasets.json). This document
reports preparation, not completed model comparisons.

Product Sentiment is retained from Phases 1–2 because it is the smallest
eligible classification dataset and already has verified historical assets.
Its earlier single-fold frozen-baseline test result is already known; it is
preserved rather than rerun or used to tune the new pilot. Video Games Sales
adds regression and numerical/categorical missing-value handling with only two
short text fields. Although it has more rows than some regression datasets,
the reduced training cap controls fitting size. Rotten Tomatoes has fifteen
features and many high-cardinality text fields; its greater encoding/TAR cost
is the reason it was not selected for this initial bounded pilot. Neither
choice depends on new predictive scores.

Both datasets belong to the twenty eligible tasks in the historical
[inventory](DATASETS.md). Their original version-1 metadata explicitly declares
`image_col: null`; no image column was removed. Metadata hashes match the Phase 1
inventory. Current release history, checked through the public Kaggle API,
lists version 1 created April 1, 2026 for the regression dataset, before the
historical source pin. The exact immutable-version request is recorded; future
changes in service availability or bytes must fail integrity checks.

| Dataset | Task | Rows | Ordered text | Ordered structured roles |
| --- | --- | ---: | --- | --- |
| `MUL_TEXT_PRODUCT_SENTIMENT` | Four-class classification | 5,091 | `Product_Description` | Numerical: `Product_Type` |
| `REG_TEXT_VIDEO_GAMES_SALES` | Regression | 16,598 | `Name`, `Publisher` | Numerical: `Year`; categorical: `Platform`, `Genre` |

Every historical feature remains available to all aligned conditions. Explicit
field order is saved rather than relying on a Python set. These declarations
match the original feature annotations and detector on the complete content
audit. Because the historical detector runs on fitting partitions, each real
comparison must also check its actual fitting-partition assignments. ReFuTab's
explicit declarations themselves do not fit vocabularies, PCA or category
statistics. All learned transforms remain training-only.

## Regression provenance and feature audit

The historical [Video Games Sales curation script](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/benchmark/datasets/REG_TEXT_VIDEO_GAMES_SALES.py)
names the [original Kaggle dataset](https://www.kaggle.com/datasets/gregorut/videogamesales).
The curated [release API](https://www.kaggle.com/api/v1/datasets/view/chico89/multabench-video-games-sales)
and [version-1 archive](https://www.kaggle.com/api/v1/datasets/download/chico89/multabench-video-games-sales?datasetVersionNumber=1)
are public and were accessed without credentials. Its metadata retains the
older source enum `REG_TEXT_SOCIAL_VIDEO_GAMES_SALES`; the curated benchmark
enum remains `REG_TEXT_VIDEO_GAMES_SALES`. This is an expected provenance
distinction, not a renamed target.

| Asset | Size | SHA-256 |
| --- | ---: | --- |
| Archive | 297,741 bytes | `2246d30847ab12d0dea5d422c09fee208a6312c0ab382db1ab8da8f4c4672567` |
| `data.csv` | 1,050,755 bytes | `635a4b91782c5d0f2f809614bfa86c66687b2ea781c90f8d6397e815a5e73d8d` |
| `metadata.json` | 210 bytes | `f364bad35542e994be2731225edb21c45233d37c800d132775f8ca41dc40f744` |
| Historical curation file | — | `bbff09b7ebbd2102ac5df35d6d5ab0d5433c18bceecef6e152e5ec80f5ae1597` |

The CSV contains exactly `Name`, `Platform`, `Year`, `Genre`, `Publisher`,
`Global_Sales`, in that raw order. `Global_Sales` is the target in the original
scale; there is no additional target transformation. `Year` has 271 missing
values and `Publisher` has 58. The other columns have no missing values, and
the target is finite. `Name`, `Platform`, `Genre`, and `Publisher` load as
objects; `Year` and the target load as float64. These are descriptive checks,
not imputation or target-driven feature selection.

There is no raw row-identifier column. Stable IDs are the original zero-based
CSV row positions and stay outside features. Game names and publisher names
are semantic benchmark features. Regional sales components and rank are absent
from the curated CSV; the explicit forbidden-feature list also prohibits
`NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Rank`, row/fold metadata,
teacher predictions and residuals. No sales component that directly sums to
the target is available to the learner.

The regression table has one duplicate complete row, two duplicate feature
rows and 5,105 game-name repetitions beyond the first occurrence. The same
game can occur on different platforms. Historical random row splitting is
retained for the benchmark comparison; it does not establish generalization
to unseen game titles. The nested diagnostic should group by exact `Name`
within the permitted outer training partition, with grouping and reduced sample
sizes recorded separately. Keep each full pipeline's PCA/gate/downstream
coordinates together; group membership is not an input feature.

Product Sentiment retains its unchanged Phase 2 schema and hashes. Its full
table has four duplicate complete rows and seven repeated descriptions; the
nested diagnostic should group exact descriptions. Original reproduction
splits and archived results remain unchanged. See the [Phase 1 audit](../results/phase1_content_audit.json).

## Preparation and integrity behavior

From the existing workspace:

```powershell
.venv\Scripts\python.exe scripts/prepare_phase4.py
.venv\Scripts\python.exe scripts/prepare_phase4.py --dataset REG_TEXT_VIDEO_GAMES_SALES
.venv\Scripts\python.exe -m unittest discover -s tests -p test_prepare_phase4.py -v
```

The standard-library [preparer](../scripts/prepare_phase4.py) verifies archive,
CSV and metadata hashes; explicit version URLs; original no-image declarations;
target/task/row/feature counts; complete column roles and exact CSV header order.
It reads only the two named ZIP members and rejects extra, duplicate or traversal
members. Every payload is validated before creating CSV/metadata files. Existing
files are checked before network access and never overwritten. Interrupted or
failed download bytes remain in `archive.zip.part` for inspection; a subsequent
attempt refuses to replace them. Complete verified files work offline.

Eight targeted tests passed in 0.113 seconds. They cover a mocked successful
download and offline replay, multiline CSV text, archive/member corruption,
existing user files and stale partials, missing/mismatched version pins,
destination containment, traversal members, original image declarations,
forbidden/overlapping roles, duplicate headers and row-count mismatches. The
real preparer successfully verified both existing local tables without fitting
or scoring a predictor. The initial regression archive download and full
content audit also succeeded; no data preparation experiment failed. A shell
quoting error during an inventory-print command was corrected before any
download or fitting; it did not modify data.

Raw archives/CSVs/metadata stay ignored below `data/`. The curated release's
declared license is “Other (specified in description)”; the original source
license was “Unknown” in the Phase 1 inventory. This preparation does not
establish redistribution permission. Only hashes, schema and aggregate audit
information are published. No new model fit, target score, GPU run or final
test prediction is claimed here. Pilot run configuration and execution status
belong in the phase protocol and handoff.
