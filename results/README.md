# Reviewed experiment summaries

`phase1_product_sentiment_frozen.json` is the reviewed real original Joint Frozen
result (macro OVR AUROC 0.8393533369478667). `phase1_content_audit.json` independently
checks splits and exact duplicate content. This is one reduced smoke task/fold;
no ReFuTab, TAR, or full-benchmark result exists.

`phase2_product_sentiment_teacher.json` records the structured-only CatBoost
teacher smoke with 900 OOF/100 validation predictions and zero test predictions
or scores. Its metrics are teacher diagnostics, not final improvement evidence.
The original baseline result is preserved; raw teacher models and row audits
remain private under ignored `results/raw/`.

Commit only reviewed lightweight summaries, tables, figures, and manifests.

`phase3_synthetic_binary_fusion.json` records a real frozen-E5 synthetic
architecture run: 60 fitting/12 validation/12 reserved test rows, CPU, 47.227 s,
and no test predictions/scores. `phase3_inference_audit.json` checks artifacts
and cold saved-bundle CSV prediction equivalence. Synthetic diagnostics do not
establish benchmark improvement. Training/checkpoint/teacher/bundle files are
private and ignored; the inference bundle has no teacher or temporary head.
Keep raw IDs, text, predictions, PCA states, datasets, and weights in ignored
`results/raw/` or private persistent storage. Record failures and incomplete runs.
