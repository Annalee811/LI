# Manuscript registration scripts: current 18 positions

This entry point reproduces the current original RGB / common-region / detail figures and registration QC. It uses the frozen, semi-automatic geometry selected for the manuscript, including the unverified coarse T2/T3 bridge at Q1-1-Z2. It does not claim a fresh fully automatic localization, retrain RoMa, or evaluate root identities. Q1-3-Z3 is excluded; 0353-1-Z2 is a legacy source alias of Q1-1-Z2 (tube Q1-1).

The accompanying ZIP preserves this repository layout. Work from its root. Use Python 3.12. Install the pinned `requirements.txt`; the matching runtime and vendor source are supplied under `analysis-scripts/all_series_localization_v2`. The scripts detect CUDA automatically; the published run used CPU. RGB/figure rebuilding does not load neural weights.

```bash
python -m pip install -r requirements.txt
python analysis-scripts/registration_manuscript_v1/workflow.py prepare --work /path/to/work --raw-root /path/to/originals
python analysis-scripts/registration_manuscript_v1/workflow.py render --work /path/to/work
python analysis-scripts/registration_manuscript_v1/workflow.py figures --work /path/to/work
```

`--raw-root` must contain `<position>/T1.ext` etc. Canonical Q1-1-Z2 and its legacy original-folder name are accepted as input sources. All 66 original file hashes are checked; unknown versions are rejected. Missing acquisitions stay absent. Omit `--raw-root` to download and verify the original V-.zip and corrected Q1.zip release archives. Those historical source archives contain excluded material, but the active working view contains only the current 18 positions.

`render` reconstructs all 66 registered RGBs using one global similarity per observation band and checks their exact published SHA256 values. T1 is a native slice; other dates use one bilinear sampling from the original. It writes `catalog.json`, metadata, native geometric masks and RGB. No dense/nonrigid flow is applied to RGB. Disjoint observation bands are separated by 24 non-data rows.

`figures` writes four single-frame, 600 dpi, A4-landscape TIFFs in `work/figures`: B3 alone (full original, upper common area, detail), other 0353 positions (5), A017 (6), Q1 (6). Detail coordinates are identical across available dates; boxes on originals are mapped by the saved transforms. Local PNGs and JSONs are review/verification artifacts, not additional requested manuscript figures. Input pixels are not enhanced or recolored. Pixel bars are T1 pixels, not millimetres.

To recompute the dense QC fields using official frozen RoMa weights:

```bash
python analysis-scripts/registration_manuscript_v1/workflow.py confidence --work /path/to/work
python analysis-scripts/registration_manuscript_v1/analyze_tables.py --tables /path/to/work/tables --output /path/to/work/summary.json
```

The full outdoor RoMa uses 560 coarse / 864 output resolution, symmetric prediction and seed 20261019. Weight URLs and SHA256 pins are in `model_links.json` and the original `model_provenance.json`. The two official weights total about 1.66 GB and are not duplicated in this small code archive. First inference downloads them through the official model loader. Device/provenance are recorded for every input pair. Float16 dense-field storage and float32 transport/accumulation match the source pipeline; GPU/device/library changes may affect inferred fields, so do not assume bitwise neural reproducibility on every platform.

Cumulative score is the product along the transported path of the minimum forward/reverse score at each step. Pixels leaving their own observation band are zeroed. Strict correspondence support requires every step to have c>=0.5, displacement<=15 T1 pixels, cycle<=10 T1 pixels and remain in the band. Low-score area means cumulative product<0.5; these two masks are not complements. These are operational QC choices, not literature-established accuracy cutoffs or calibrated probabilities. Geometric masks retain changing roots and are distinct from correspondence masks.

`published_tables/` in the ZIP is the immutable table snapshot used for the supplied results/discussion, not a newly recomputed confidence run. Reproduce its descriptive summaries with `analyze_tables.py --tables published_tables --output summary.json`. Series comparisons average within each position first. Cumulative stage comparisons use only the same 12 complete four-date positions. These are descriptive statistics, not independent tube-level significance tests.

## Localization source and limitations

The original matching/RootSIFT/RANSAC/multi-route selection implementation remains in `analysis-scripts/all_series_localization_v2`, with the source-specific root-band recheck in `analysis-scripts/0353_1_Z2_rootguided_v3`. Their old paths/IDs are retained for audit. The current wrapper and registry define the manuscript identities. Do not run the historical 19-position `reproduce.sh` as the current active entry point: it intentionally reproduces an older snapshot.

Some previously reviewed geometries were reused only after original SHA256 checks. Revised Q1 searches use grayscale CLAHE, blurred and elongated-ridge views for feature extraction only, RootSIFT mutual ratio 0.84, width-1600 analysis, RANSAC threshold 5 analysis pixels / 20,000 iterations / confidence 0.999. The methods differ in source-specific manual bands and evidence routes; they are not a single completely automatic training pipeline. Q1-1-Z2 includes a user-region/ridge-correlation coarse translation, with no claim of successful RoMa point correspondence for that bridge.

The current registry preserves the source-specific evidence, transforms and review flags. No manual landmark TRE, root-identity validation, or new model training is supplied. Area coverage and RoMa scores are not registration accuracy. Root-system segmentation weights from U-Net, DeepLabV3+ and SegFormer belong to a different analysis and are not substituted for the RoMa model here.

Third-party code retains its upstream license. References in `CITATION.bib` support RoMa, RootSIFT, RANSAC and registration methodology; our thresholds/masks are explicitly custom definitions. No article DOI or software DOI has been invented. Use the final immutable GitHub commit as the code provenance in a manuscript; if depositing to an archive, obtain the archive's real DOI after upload.
