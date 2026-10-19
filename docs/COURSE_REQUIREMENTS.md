# Course requirements

Reviewed on **October 19, 2026** against all eight pages of the official local PDF, including visual inspection of every rendered page. The PDF is authoritative for academic requirements; project preferences below are identified separately.

## Source and provenance

- Course: Technion NLP (097215), Spring 2026, *Final Project: Text-Tabular Learning*.
- Local file: `NLP Course (097215) Final Project_ Text-Tabular Learning.pdf` in the workspace root.
- SHA-256: `406ca54fc55e77d54058b0b14fa32ffddf3bb24d5a41ddc69ac08880399a8811`.
- Page references in this document use the PDF's printed pages 1–8.
- The PDF and its extracted text/rendered pages are local reference material. Do not commit or publish them without establishing redistribution permission. This document records requirements rather than redistributing the source PDF.

## Selected track and comparison target

The user selected **Track 1: Algorithm**. Its objective is to design and implement an algorithm for classification and/or regression on tabular datasets with free-text features, evaluated on MulTaBench (§1.1 and §2.1, p. 2).

| Use of targets to build text representations | Required comparison (§2.1, p. 2) |
| --- | --- |
| No target use | Demonstrate statistically significant improvement over **Joint Frozen**. |
| Target use, through encoder fine-tuning **or any other way** | Outperform **Joint TAR**, the target-aware representation baseline. The track summary describes statistically significant improvement over the relevant baseline as the success objective (§1.1). |

ReFuTab's residual-trained fusion representation uses labels even while E5 is frozen. Therefore **Joint TAR is the primary baseline**. A frozen encoder does not exempt the supervised fusion module from the target-aware comparison. Joint Frozen remains a useful additional comparison.

Creativity, motivation, and deep analysis are valued even when an approach does not decisively outperform the baseline. Merely replacing the encoder with a stronger model is not an interesting contribution on its own (creativity box, pp. 2–3). This is guidance about assessment, not permission to claim unmeasured improvements.

## Suggested directions (§2.2, p. 3)

The course suggests tabular-aware text encoders or input verbalization, novel embedding fusion, and realistic synthetic text-tabular data augmentation. These are examples, not a prescribed architecture. ReFuTab fits the smart embedding fusion direction; the PDF does not prescribe residual guidance, CatBoost teachers, fold counts, dimensions, gates, or specific losses.

## Leakage requirement and unresolved wording (§2.2, p. 3)

The complete leakage box states:

> You must ensure that your training or pretraining phase does not leak test set information. If you use data from a MulTaBench dataset during training/fine-tuning, you must evaluate that specific dataset using a model variant that has never seen it during its training process. For instance, you should train separate models (or use cross-validation) to evaluate each dataset in the benchmark cleanly.

The prohibition on test-information leakage is explicit. The phrase “that specific dataset” and “has never seen it” is ambiguous: read literally, it could require excluding the entire evaluation dataset from representation training; the examples of separate models/cross-validation can also support ordinary supervised fitting on permitted training partitions. The PDF does not resolve that tension.

**Clarification is required before choosing the target-aware evaluation protocol.** Do not silently treat normal dataset-specific supervised fitting as a confirmed interpretation of the literal requirement. Pending clarification, maintain separate held-out data, train-only preprocessing and representation fitting, isolated model variants, no test-driven selection, and complete split provenance. The user's OOF teacher and cross-fitting safeguards strengthen the protocol but do not by themselves resolve this wording. See [DATA_LEAKAGE.md](DATA_LEAKAGE.md).

## Track 1 submission checklist (§2.3, p. 3)

- [ ] **Source code:** general-purpose implementation executable on an arbitrary text-tabular dataset, packaged with pinned dependencies. The PDF gives `pyproject.toml` plus `uv.lock` as an example, while permitting equivalent conventions.
- [ ] **Evaluation results:** performance reports across **all datasets in MulTaBench**, using **AUROC for classification** and **R² for regression**. Dataset modality scope and multiclass AUROC details require clarification below.
- [ ] **Pretrained/fine-tuned model artifacts, when applicable:** upload model weights to **Hugging Face**. Code may load public weights directly; the PDF says a dedicated download script is unnecessary.
- [ ] **Inference mode:** run the algorithm using the uploaded models.
- [ ] **End-to-end training mode:** train/fine-tune the models from scratch to reproduce the uploaded weights.
- [ ] **Final report:** the English short paper specified in §4.

ReFuTab uses a pretrained encoder and will learn fusion artifacts, so plan for the model-artifact and two-mode deliverables. The precise artifact inventory, including whether already-public unchanged E5 weights need re-uploading, is not defined in the PDF; document references and learned weights explicitly rather than claiming this is settled.

## Track 2 requirements are distinct

Track 2 curates new datasets and is **not the selected project track** (§3, pp. 4–6). Its curation procedure explicitly uses official MulTaBench code, **five learners** (TabM, CatBoost, LightGBM, TabPFN 2.0, TabPFN 2.5) and **four conditions** (structured only, unstructured only, Joint Frozen, Joint TAR). For at least three of five learners, Joint Frozen must strictly exceed both unimodal baselines and Joint TAR must strictly exceed Joint Frozen (§3.2–§3.2.1, p. 4).

Its standard scope is one new qualifying dataset; the optional outstanding scope uses a systematic generation/mining method with at least five qualifying datasets (§3.3, pp. 5–6). Deliverables include curated CSVs, creation/evaluation code, a five-model/four-condition results CSV, and the report (§3.4, p. 6). The staff recommends starting with the standard scope.

**Do not automatically import Track 2's five-learner/four-condition requirement into Track 1.** Track 1 requires comparison with its relevant baseline and all-dataset evaluation; the appropriate downstream learner coverage must be established from the historical benchmark protocol and any staff clarification. ReFuTab's four planned comparisons are a user research design, with standard gated fusion as an additional ablation.

## Final report (§4–§4.1, p. 6)

- English short academic paper using the **NeurIPS LaTeX template**.
- **At most four pages**, excluding references and appendices.
- Appendices may contain **result tables/metrics only**; do not move method explanations or other prose there to evade the page limit.
- Required sections: **Title & Abstract; Introduction; Related Work; Method / Dataset; Results; Conclusion & Summary**.
- For Track 1, describe the algorithm's architecture and design in Method. Cite relevant literature, including the models used and MulTaBench. Present experimental setup, baselines, and performance metrics using tables/charts. Discuss limitations and future directions in the conclusion.

## Submission, deadline, and conduct (pp. 1–2, 7)

- Work is **individual, with no pairs**.
- Submit **all deliverables via the course Moodle** by **Monday, October 26, 2026**.
- **No clock time or deadline timezone is stated in the PDF.** Verify the Moodle submission setting; do not invent a 23:59 deadline.
- Late submission costs **20 final-grade points per day of delay** unless covered by Technion School of Undergraduate Studies regulations (submission box, p. 7).
- AI tools are permitted and encouraged, and the student is responsible for everything submitted (p. 1 and §1.2, p. 2).
- Advisor: **Eilam Shapira**; the project Moodle forum is the stated support channel, with staff aiming to answer within one week (§1.2, p. 2). No message has been sent on the user's behalf.

## Conflicts and questions to resolve before implementation

| Issue | Verified source distinction | Required disposition |
| --- | --- | --- |
| **All datasets versus modality scope** | §2.1 concerns tabular data with free text; §2.3 literally requires results across all MulTaBench datasets. MulTaBench also includes image modalities. The PDF supplies no Track 1 exclusion rule. | Identify the pinned dataset inventory and obtain the intended handling of image-only/image-containing tasks. Do not silently narrow “all datasets” to text-only tasks. |
| **Classification metric** | §2.3 requires AUROC for classification but does not define multiclass averaging or absent-class handling. The pinned historical source uses macro one-vs-rest AUROC with a present-class fallback; current master switched to multiclass log loss. See [UPSTREAM.md](UPSTREAM.md). | Validate the historical convention and document class-subset behavior before finalizing evaluation; retain the course's AUROC requirement. Do not import current log loss or substitute accuracy silently. |
| **Training on an evaluation dataset's training partition** | The leakage paragraph's literal dataset exclusion wording is stronger than ordinary split-level holdout and ambiguous alongside its examples. | Ask for the intended rule and record the response; OOF teacher predictions alone do not settle it. |
| **Paper identifier** | PDF reference [1], p. 7, labels MulTaBench as `arXiv:2605.12292`. That identifier actually resolves to [STRABLE](https://arxiv.org/abs/2605.12292). The user-provided [arXiv:2605.10616](https://arxiv.org/abs/2605.10616) resolves to MulTaBench. Both destinations were verified during Phase 0. | Preserve this confirmed citation error in provenance; use the verified MulTaBench link for research references. |
| **Historical repository date** | **July 26, 2026** is a user/tutor-provided course-publication reference date, not a publication date stated by the PDF. The PDF lists a **July 19, 2026 lecture**, a separate event (p. 1). | Use the tutor's July 26 instruction for historical `master` investigation, without attributing that date to the PDF. Record the selected SHA and any ambiguity separately. |

These unresolved points do not prevent Phase 0 documentation and repository setup. They prevent treating an unconfirmed experiment protocol as an established academic requirement.

## User research choices, not course mandates

The user selected Track 1, the ReFuTab residual-guided fusion hypothesis, Option A before Option B, a three-fold OOF CatBoost teacher, frozen E5-small-v2, training-only PCA to 30 dimensions, a compact structured encoder, an approximately identity-initialized gate, a temporary correction head, downstream learner retraining, and the standard-gate ablation. The suggested hidden width of 64 and dropout near 0.1 are initial hyperparameters.

GitHub synchronization, phased implementation, persistent handoff files, Colab workflow, repeated-run variability, runtime reporting, and additional leakage tests are user project requirements. They support reproducibility but are not stated as course-specific mandates. The course does not guarantee success for this architecture or require Option B.
