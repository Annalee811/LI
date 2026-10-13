# Reproduce 0353-3 same-observation-region outputs

The original inputs are from the v-change-data GitHub release. Place the original folders under this evidence directory's `input/raw/V-变化/`. Input hashes are checked against saved manifests. Do not substitute exports from prior workflows.

The reusable scripts and helpers are in the parent analysis-scripts directory. Published evidence contains cached RoMa regional candidates, complete-frame RootSIFT feature searches, correspondence CSVs, global similarity matrices, separate conditional point-CV and block-fit pose stability, and unused-edge route consistency checks. RoMa supplies broad location hypotheses; descriptors, root-structure proxies and spatial coverage weight the global fit. Root-support candidates are not confirmed root identities.

From repository root with the existing RoMa Python environment:

```bash
OPENBLAS_NUM_THREADS=1 python analysis-scripts/export_same_region.py --root analysis-scripts/0353-3_same_region_evidence --position 0353-3-LB2 --threads 1
```

Feature evidence can be regenerated with search_same_region.py, same --root/--position, --width 1000. Positions requiring refined searches also append --width 1600 --append; metadata/full_canvas_feature_search.json records the actual widths used. Export only applies global scale, rotation and translation; each original RGB image is sampled once onto the T1 scale. Detection preprocessing never changes saved RGB. Crops use the largest horizontal all-period geometric observation rectangle, not a mask asserting every point confidently matches.

Point-CV is conditional on descriptor-consensus anchors and not independent ground truth or manual TRE. Spatial fold pose stability measures sensitivity to blocks of selected anchors, not absolute accuracy. The 3-pixel point-pass flag is retained separately from region stability. Region stability uses >=20 anchors, >=30% horizontal coverage, fold pose median<=3/p95<=5 T1 px, and point-CV median<=10 T1 px. At least one unused structural edge must corroborate the selected graph within 5 T1 px. All eligible unused-edge disagreements are reported, including failures. Per-position README and localization_metadata.json state the remaining limitations; no new-root/death classifications are inferred. Positions with <=2 available periods are skipped; missing periods are not synthesized.
