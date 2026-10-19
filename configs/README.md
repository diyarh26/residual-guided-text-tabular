# Configurations

`datasets.json` records the 20 eligible historical schemas and 20 image-column
exclusions. Only the Product Sentiment pilot CSV has been fetched and hashed.
`phase1_assets.json` pins source inputs and model-file hashes; `phase1_frozen.json`
fixes the completed CPU smoke baseline. The configuration was committed before
test evaluation. `phase2_teacher.json` fixes a small, untuned structured-only
teacher budget and reconstructs those same historical splits from the pinned
source and public Phase 1 fingerprints. It does not score the outer test.
Other datasets require audited explicit schemas and versioned file hashes;
generic runner protocols also support saved IDs or declared local splits.
Fusion configurations remain Phase 3 work.
