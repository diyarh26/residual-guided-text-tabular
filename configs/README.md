# Configurations

`datasets.json` records the 20 eligible historical schemas and 20 image-column
exclusions. Product Sentiment and Video Games Sales CSVs are fetched and hashed.
`phase1_assets.json` pins source inputs and model-file hashes; `phase1_frozen.json`
fixes the completed CPU smoke baseline. The configuration was committed before
test evaluation. `phase2_teacher.json` fixes a small, untuned structured-only
teacher budget and reconstructs those same historical splits from the pinned
source and public Phase 1 fingerprints. It does not score the outer test.
Other datasets require audited explicit schemas and versioned file hashes;
generic runner protocols also support saved IDs or declared local splits.
`phase3_smoke.json` fixes the synthetic binary architecture smoke before its
diagnostics: two text fields, 60 fitting/12 validation/12 reserved test rows,
real frozen E5, five gate epochs and a 100-tree downstream LightGBM. Other
configurations may use the generic CSV schema/split interface. This is not a
Phase 4 benchmark or matched-baseline pilot configuration.

`phase4_datasets.json` pins both original no-image schemas, archive/member
hashes and data audits. `phase4_pilot.json` registers four matched methods,
three historical outer holdouts per dataset, grouped nested diagnostics and
the descriptive decision rule before new pilot scores. Neither configuration
claims the required GPU comparisons have executed.
