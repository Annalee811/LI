# v9 manuscript re-analysis for Plant Phenomics

This package returns to the user's explicitly selected **v9 binary root/background** experiment, commit `608659941cd720a8292efbebda782e5277e30f08`. It does not use v13 temporal annotations, train models, adjust thresholds, redraw roots or change the original masks. All seven original archives are saved again byte-for-byte. The original user progress JSON SHA256 is `27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09`.

C0–C2 train; C3 selects models and parameters; C4 is a repeatedly inspected same-position benchmark. All four dates in one spatial column stay together. Four original RGB images from one annotated location yield12/4/4 date-region views and **zero independent external test images**. Spatial grouping avoids direct overlap but cannot establish cross-position or cross-tube generalization.

The recommended primary pilot score is **C4 four-date pooled pixel counts**, with C3 separately reported. Four-date means±SD are descriptive, not biological confidence intervals. C3/C4 arithmetic averaging, pooled pixels and the mean of eight date views answer different questions; mixed-region results are exploratory only and cannot replace the main C4 evaluation. R² is identity agreement, not squared correlation, and references are mask-derived pixel skeleton lengths, not independent manual length measurements. Every image/period is retained, including errors and reductions.

## Contents and reproducibility

The analysis-data ZIP contains `inputs/`, `source/v9_manuscript/`, `source/original_v9/`, and the source-archive manifest. This analysis requires Python3.12 plus requirements.txt; no GPU or training is needed. Original model-training/inference source, TF runtime/vendor source and unchanged model weights remain in the re-saved original result bundles. Their existing automaticCUDA behavior and original TF1.12 environment are preserved.

From the analysis-data ZIP root:

```bash
python -m pip install -r source/v9_manuscript/requirements.txt
python source/v9_manuscript/run_all.py --inputs inputs --output rebuilt
```

In the current cloud environment, dependencies were installed with `UV_CACHE_DIR=/tmp/B3_v9_uv_cache uv pip install --python .venv/bin/python python-docx==1.2.0`; native statistics, all plots and DOCX generation were actually executed. Use the existing checkout; no worktree or web service is needed.

To reconstruct inputs from the original full result bundles in the repository:

```bash
python export_inputs.py --run results/B3_three_model_v9 --data results/B3_root_background_v5 --archives roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-continuity-v9 --output NEW/inputs
python verify_original_files.py --inputs NEW/inputs --archives ORIGINAL_ARCHIVE_DIRECTORY --output NEW/original_file_verification.json
```

The exporter verifies all seven archive SHA256 values and the original archived metric JSON. The separate verifier checks57 native RGB,mask,probability andannotation files byte-for-byte against the original ZIPs. analyze.py recomputes24 native evaluation masks, per-period and pooled TP/FP/FN/TN metrics,192 correlated block-date/model lengths,and exact raw-score average precision; all agree with v9 source results. Models and original human labels remain unchanged.

Ten figure groups: Fig1 design,Fig2 C4 metrics,Fig3 C4 fixed-field errors,Fig4 three-model C4 length regression/Bland–Altman; FigS1 C3 metrics,S2 C3 length,S3 temporal/spatial heterogeneity,S4 different aggregation methods,S5 PR/AP,S6 four-date mean±SD. Export600dpi PNG/TIFF andPDF; quantitative plots also have SVG. Optical detail is limited by original images, not thedpi metadata. Full English captions and editable Chinese/English Word documents explain how to use the assets.

## Journal-specific status

Target: Plant Phenomics. On2026-10-18,the official journal and author-guideline URLs could not be accessed because the environment's outbound domain policy rejected HTTPS CONNECT. We do **not** claim to have verified current journal-specific dimensions,formats,limits,fees or submission fields. The package uses general high-resolution preparation defaults and supplies official URLs and a list of items to confirm. No manuscript has been submitted, noDOI invented and no source/reference label altered. The manuscript DOCX is an editable Methods/Results draft, not a complete ready-to-submit paper; acquisition details,scientific context and author records must be supplied from actual records.

Suggested manuscript placement: mainFig1–4 and C4Tables1–2; C3/per-period/PR/mean summaries in supplements. The study remains a single-position pilot, with moderate actual segmentation performance. It does not demonstrateDice>.9,significant model superiority or external generalization.
