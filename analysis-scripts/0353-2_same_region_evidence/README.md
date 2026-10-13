# Reproduce 0353-2 same-observation-region outputs

Original image inputs are the v-change-data release, with input SHA256 checked against the cached manifest. Unpack original folders under this directory's `input/raw/V-变化/`; do not substitute the PNG/JPEG exports from earlier workflows. Both feature search and export accept `--root` pointing to this evidence directory and `--position`.

Published evidence includes hash-verified cached RoMa point candidates, complete-frame RootSIFT searches, automatic correspondence CSVs, global similarity matrices and numerical limitations. Cached RoMa proposes broad region hypotheses; independent descriptor matching, root-structure proxies and spatial coverage weight the global fit. A root proxy is not a confirmed root identity.

From repository root, with the existing RoMa Python environment:

```bash
OPENBLAS_NUM_THREADS=1 python analysis-scripts/export_same_region.py --root analysis-scripts/0353-2_same_region_evidence --position 0353-2-LB2 --threads 1
```

To rebuild feature evidence rather than reuse saved searches, run search_same_region.py at --width 1000; LB2/LB3 also use --width 1600 --append. Then run export. Dependencies/helper files reside beside these scripts. Original RGB is sampled once with a uniform-scale/rotation/translation map; T1 pixels are unchanged within the crop. Detection preprocessing does not alter saved RGB.

Spatial fold pose stability is sensitivity to blocks of selected anchors, not absolute registration accuracy or independent ground truth. Point CV uses descriptor-consensus anchors, is conditioned on that selection and is not manual TRE. The original 3-pixel point-pass flag is retained. Regional stability is evaluated separately (20 anchors, 30% horizontal coverage, block-fit pose median<=3/p95<=5 T1 px, point median<=10 T1 px); one unused edge must corroborate a selected region graph within 5 T1 px. All unused-edge disagreements are reported even if they fail. LB2's T3 routes fail this check (median 6.38/9.95, p95 12.66/13.06 T1 px) and require visual review. No claim of full consecutive point registration success is made. Missing images are not synthesized; positions with <=2 images are skipped.
