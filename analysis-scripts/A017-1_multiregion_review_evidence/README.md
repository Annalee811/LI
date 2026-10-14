# Reproduce A017-1 available-period regional review

The two positions contain only original T1/T2/T4 RGB, missing T3. Put release originals under `<position>/input/raw/V-变化/<position>` in this evidence root and preserve saved hashes. Link each band's parent's `input` and position-named RoMa cache into `band_A`/`band_B`/`band_C` as applicable. Published evidence retains RoMa proposals, all complete-frame and region-conditioned descriptor candidates, prior matrices, accepted and rejected correspondence checks, final crop matrices, root-window coordinates and all failures.

From repository root with configured Python:

```bash
OPENBLAS_NUM_THREADS=1 python analysis-scripts/export_multiregion_band.py --root analysis-scripts/A017-1_multiregion_review_evidence/A017-1-B2/band_A --position A017-1-B2 --prior-matrices analysis-scripts/A017-1_multiregion_review_evidence/A017-1-B2/band_A/prior.json --threads 1 --allow-no-extra-route --allow-inconsistent-routes
OPENBLAS_NUM_THREADS=1 python analysis-scripts/render_available_period_regions.py --root analysis-scripts/A017-1_multiregion_review_evidence/A017-1-B2 --position A017-1-B2 --bands A B C --windows analysis-scripts/A017-1_multiregion_review_evidence/A017-1-B2/root_windows.json --threads 1
```

Export B2 A/B/C individually before rendering. Export Z2 only band_B; renderer receives --bands B. Z2 band_A export fails the validated graph test even after extra crack/gradient searches, and no crop for it is allowed in accepted output. Its failure evidence is retained, not bypassed. Existing Z2 T1/T2 cached RoMa has zero reliable candidates; the regional bridge is structurally proposed and independently corroborated, never called a direct neural success.

Whole-frame searches: original width1000 CLAHE/blur, then appended width1300 CLAHE/blur/ridge for both positions; Z2 additionally has width1300 crack/gradient. search_available_period_regions.py performs guidance inside common-band proposals with actual period IDs and80 canonical pixel mask expansion. Final fits restrict anchors to the common-band proposal plus20 canonical pixels and retain all disagreements. These checks are conditional on region proposals, not independent ground truth.

Full RoMa automatically detects CUDA; this run used CPU. Added full T1/T4 supported candidate counts: B2=1866, Z2=1167. One dominant RoMa mode does not exclude another structurally supported band. Dense neural warps never render final RGB. Region output uses only global uniform scale/rotation/translation; original RGB is sampled once. The renderer verifies input SHA256, exact T1 crops, source bounds, equal within-region saved dimensions, uniform similarity singular values, lossless PNG encoding and ZIP CRC. B2 has24-pixel non-data separators; Z2 has none. Geometric masks are not confidence or effective point masks. Missing T3 is never synthesized, omitted source gaps do not establish root continuity, root identities and life/death are not inferred, and actual manual TRE remains unassessed.
