# Model artifacts

This folder is ignored except for this README. Store checkpoints, PCA states,
encoders, and downstream models in a persistent artifact store with run IDs,
source/config hashes, schema/class ordering, and appropriate permissions.
Phase 1 stores the pinned frozen E5 snapshot here. Its provenance and file hashes
are verified by preparation and the baseline runner. Raw baseline PCA outputs
remain under ignored `results/raw/`; no ReFuTab weights exist yet.
