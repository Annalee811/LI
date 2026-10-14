# Reproduce the 0353-3-LB3 two-region review

The evidence keeps both disjoint branches and all route failures. Original RGBs are from the v-change-data release; their SHA256 hashes must match the saved manifest. Put originals under `input/raw/V-变化/0353-3-LB3` in this evidence directory. No physical periodic/cylindrical wrap is assumed.

Prepare each `band_A`/`band_B` workspace by copying `full_canvas_feature_search.json` into its `0353-3-LB3_region_localization` directory, linking the shared `input` and `0353-3-LB3` cache directories from its parent. Keep the saved branch prior.json. Then from repository root run with the configured RoMa Python environment:

```bash
OPENBLAS_NUM_THREADS=1 python analysis-scripts/export_LB3_band.py --root analysis-scripts/0353-3-LB3_two_region_review_evidence/band_A --position 0353-3-LB3 --prior-matrices analysis-scripts/0353-3-LB3_two_region_review_evidence/band_A/prior.json --threads 1
OPENBLAS_NUM_THREADS=1 python analysis-scripts/export_LB3_band.py --root analysis-scripts/0353-3-LB3_two_region_review_evidence/band_B --position 0353-3-LB3 --prior-matrices analysis-scripts/0353-3-LB3_two_region_review_evidence/band_B/prior.json --threads 1
OPENBLAS_NUM_THREADS=1 python analysis-scripts/review_0353_3_LB3_two_regions.py --root analysis-scripts/0353-3-LB3_two_region_review_evidence --threads 2
```

Full-frame searches include width1000 CLAHE/blur plus width1600 CLAHE/blur/ridge/gradient. Recreate searches with search_Q1_region_structures.py; keep 1000-width search with --modes clahe blur, then append 1600-width four modes. collect_roma_region_candidates.py --extra-pair T1_to_T4 regenerates the added complete-RoMa check if its cached folder is removed from a separate reconstruction workspace; CUDA is automatically detected. The original cached adjacent RoMa evidence is preserved.

Rendering verifies original hashes, exact T1 crop pixels, equality of the two linear singular values (global uniform similarities only), source image bounds, PNG encoding and ZIP CRC. Each region is sampled only once from original RGB. Observational masks are geometry only, not neural confidence or effective point support. The joined image has an explicit 24-pixel non-data divider. The 395-pixel original-T1 gap is never fabricated or treated as continuous root data. A region remains a review candidate because its direct T1/T4 route disagrees by 6.65 T1 pixels despite other corroborating routes. No true manual TRE, root identities or root life/death labels are asserted.
