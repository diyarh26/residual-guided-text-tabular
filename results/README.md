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
Keep raw IDs, text, predictions, PCA states, datasets, and weights in ignored
`results/raw/` or private persistent storage. Record failures and incomplete runs.
