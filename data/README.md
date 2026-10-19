# Local data

Datasets and splits with sensitive identifiers remain outside Git. This folder
is ignored except for this README. Record dataset revisions, permissions, schema,
split fingerprints, and approved artifact locations in experiment manifests.
Phase 1 stores the version-1 Product Sentiment pilot and schema-audit metadata
locally here; a public clone fetches them using `scripts/prepare_phase1.py`.
No raw dataset is redistributed in Git. Other eligible CSVs are not downloaded.
