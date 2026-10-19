# Historical dataset inventory

Audited October 19, 2026. This is Phase 1 provenance and schema analysis, not a
completed evaluation or approval of every feature. The machine-readable roster
is [configs/datasets.json](../configs/datasets.json).

The user selected datasets whose original benchmark schema has **no image
column**. At the historical MulTaBench pin
`599bd6a5631c96f8aef297cc4cb6e4c197ae0dca`, the curated
[`MulTaBenchDatasetID` roster](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/datasets/all_datasets.py)
has 40 datasets. Its 20 text datasets are eligible and its 20 image datasets
are excluded. The other source/candidate enums are not additional curated
benchmark tasks.

Eligibility was checked against all 40 original Kaggle version-1
`metadata.json` files: every eligible task has `image_col: null`; every excluded
task names an image column. These declarations agree with the pinned text
curation scripts, which call `save_dataset` without its `image_col` argument.
The default is `None` in the pinned
[`save_dataset` helper](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/benchmark/utils/curation.py).
No image column was removed to make a dataset eligible.

Every eligible Kaggle release has only version 1 in the checked version
history, created April 1, 2026, before the selected source commit. Inventory
entries pin this version and metadata SHA-256. Only the small pilot's data CSV
was downloaded and hashed; the other 19 CSVs still require content hashes before
execution. A metadata hash is not a hash of its underlying table. The later
typed Hugging Face migration was not substituted for these historical releases.

| Dataset ID suffix | Task | Target | Rows | Features | Classes |
| --- | --- | --- | ---: | ---: | ---: |
| `FAKE_JOB_POSTING` | Binary | `fraudulent` | 12,725 | 5 | 2 |
| `JIGSAW_TOXICITY` | Binary | `Is Toxic` | 100,000 | 31 | 2 |
| `KICKSTARTER_FUNDING` | Binary | `Funding Status` | 86,502 | 9 | 2 |
| `DATA_SCIENTIST_SALARY` | Multiclass | `salary` | 15,841 | 6 | 6 |
| `MICHELIN_RESTAURANTS` | Multiclass | `Award` | 18,843 | 11 | 5 |
| `PRODUCT_SENTIMENT` | Multiclass | `Sentiment` | 5,091 | 2 | 4 |
| `SPOTIFY_GENRES` | Multiclass | `track_genre` | 114,000 | 18 | 114 |
| `US_ACCIDENTS` | Multiclass | `Severity` | 100,001 | 44 | 4 |
| `WINE_REVIEW` | Multiclass | `variety` | 84,123 | 5 | 30 |
| `WOMEN_CLOTHING_REVIEW` | Multiclass | `Rating` | 18,788 | 10 | 5 |
| `BABIES_PRICES` | Regression | `price` | 5,085 | 12 | - |
| `BOOK_PRICE` | Regression | `Price in USD` | 4,989 | 8 | - |
| `BOOK_READABILITY` | Regression | `New Dale Chall Readability Formula` | 4,724 | 30 | - |
| `MERCARI_MARKETPLACE` | Regression | `log_price` | 100,000 | 9 | - |
| `MONTGOMERY_SALARIES` | Regression | `current_annual_salary` | 9,228 | 11 | - |
| `SCIMAGOJR_IMPACT` | Regression | `H index` | 31,136 | 22 | - |
| `ROTTEN_TOMATOES` | Regression | `RatingValue` | 7,158 | 15 | - |
| `VANCOUVER_SALARIES` | Regression | `remuneration` | 44,574 | 5 | - |
| `VIDEO_GAMES_SALES` | Regression | `Global_Sales` | 16,598 | 5 | - |
| `ZOMATO_RESTAURANTS` | Regression | `rate` | 41,665 | 15 | - |

Full enum names have `BIN_TEXT_`, `MUL_TEXT_`, or `REG_TEXT_` prefixes as recorded
in the JSON. The JSON also preserves each pinned source annotation's ordered
feature names and pandas dtypes. These are schema evidence, not the runtime
text/category assignments: the historical detector determines text fields from
the permitted fitting partition. Some `metadata.dataset_id` values retain older,
longer source-enum names; the selected benchmark enum and slug govern loading.

## Small baseline pilot

`MUL_TEXT_PRODUCT_SENTIMENT` is recommended: the smallest classification task in
the eligible roster, one text field, one structured field, and four classes.
It exercises multiclass AUROC without the many fields or long descriptions of
larger tasks. Its official
[Kaggle release](https://www.kaggle.com/datasets/chico89/multabench-product-sentiment)
and [release API](https://www.kaggle.com/api/v1/datasets/view/chico89/multabench-product-sentiment)
are public; the version-1 CSV and metadata were downloaded without credentials.

| Artifact | Exact reference |
| --- | --- |
| Curated release | `chico89/multabench-product-sentiment/versions/1` |
| CSV download | [Version-1 data.csv](https://www.kaggle.com/api/v1/datasets/download/chico89/multabench-product-sentiment/data.csv?datasetVersionNumber=1) |
| CSV local path | `data/multabench-product-sentiment-v1/data.csv` (ignored) |
| CSV SHA-256 | `8a75a9880a5a1a619b16622c7f8f257ff8d19fde9b02864bf0059d5b0a8c2bb7` |
| CSV bytes | 604,118 |
| Metadata SHA-256 | `db78538f386c3eed11df6c9110fbd8de30c9f7c72311ff85ee6811fd75bfde13` |
| Original source | [OpenML 46651, version 1](https://www.openml.org/api/v1/json/data/46651) |
| Source curation | [Pinned Product Sentiment script](https://github.com/alanarazi7/MulTaBench/blob/599bd6a5631c96f8aef297cc4cb6e4c197ae0dca/multabench/benchmark/datasets/MUL_TEXT_PRODUCT_SENTIMENT.py) |

Features, in file order, are `Product_Description` and `Product_Type`. Preserve
the historical CSV interpretation of `Product_Type` as integer structured data;
do not force a categorical recoding and call it unchanged reproduction. Target
counts are `Cannot Say: 97`, `Negative: 324`, `No Sentiment: 1,653`, and
`Positive: 3,017`. All three columns have zero missing cells. Persist original
zero-based CSV row position as provenance outside the model's feature matrix.
Fit and save the target encoder/class order using permitted training rows.

There are four duplicate complete rows and seven repeated descriptions beyond
the first occurrence. The historical benchmark splits rows rather than duplicate
groups; its reproduction therefore does not establish isolation of identical
text across partitions. Record this limitation and audit a grouped diagnostic
before making new method claims. Do not change historical reproduction splits
silently. No training or score is claimed by this inventory.

## Feature and license gates for later evaluation

Image eligibility does not certify feature legality or lack of target-derived
information. In particular:

- `WOMEN_CLOTHING_REVIEW` retains `Unnamed  0`, a probable exported row index.
  Resolve its treatment under the row-ID exclusion rule before method evaluation.
- `BOOK_READABILITY` includes other readability formulas, word/sentence counts,
  and related scores. Audit which are target-derived metadata before use.
- `SCIMAGOJR_IMPACT` includes rank, related impact statistics, and identifiers;
  salary tables include names and related pay values. Record feature-purpose
  decisions before fitting the method.
- `US_ACCIDENTS` has temporal/geographic dependence and event-end fields; audit
  task timing. Spotify's 114 classes require explicit coverage checks under
  training caps and held-out macro-AUROC.

Every curated Kaggle release declares **Other (specified in description)** and
delegates rights to its original source. The JSON records source metadata
lookups separately; their license strings do not establish permission beyond
what the actual terms grant. For example, the pilot's OpenML metadata says
**MachineHack prediction competition**, which is not a standard license grant.
It does not justify redistributing raw data or embedding caches. All downloaded
tables, metadata response copies, and caches remain ignored.

Some source declarations are explicit: the
[Michelin source API](https://www.kaggle.com/api/v1/datasets/view/ngshiheng/michelin-guide-restaurants-2021)
reports CC BY-NC 4.0, the
[US Accidents source API](https://www.kaggle.com/api/v1/datasets/view/sobhanmoosavi/us-accidents)
reports CC BY-NC-SA 4.0, the
[CLEAR source API](https://www.kaggle.com/api/v1/datasets/view/verracodeguacas/clear-corpus)
reports CC BY-SA 4.0, and
[OpenML Montgomery metadata](https://www.openml.org/api/v1/json/data/42125)
reports Public Domain (CC0). Other declarations are incomplete or unspecified.
These are retrieval-time provenance records; redistribution of any dataset or
data-derived artifact still needs the applicable source terms and attribution.
Public model/learned-artifact upload is a separate later-phase decision.

Ignored audit evidence is under `data/dataset-inventory-audit/`: all 40 raw
version-1 metadata files, eligible release view responses, original-source
metadata responses, and the assembled local `audit.json`. The tracked inventory
stores hashes and URLs so the evidence can be fetched independently.
