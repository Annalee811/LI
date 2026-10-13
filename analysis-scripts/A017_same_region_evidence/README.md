# Reproduce all A017 same-observation-region outputs

Original RGB inputs come from the v-change-data GitHub release. Put original per-position folders under this evidence directory's input/raw/V-变化/. Their SHA256 values are checked against published manifests.

The parent analysis-scripts contains export_A017_same_region.py, collect_roma_region_candidates.py and reusable search_same_region.py/B2_region_tools.py helpers. Evidence preserves hash-verified RoMa candidates, model/cycle thresholds, full-frame descriptor searches, conditional spatial-block point validation and pose stability, actual global similarity matrices, and all eligible unused-edge route differences. RoMa candidates propose broad regions; descriptors, root-ridge proxies and spatial coverage weight a global fit. The root proxy is not validated segmentation.

Example from repository root:

```bash
OPENBLAS_NUM_THREADS=1 python analysis-scripts/export_A017_same_region.py --root analysis-scripts/A017_same_region_evidence --position A017-2-2 --threads 1
```

Where metadata contains coarse_hypothesis_override, also supply --prior-matrices analysis-scripts/A017_same_region_evidence/<position>_region_coarse_hypothesis.json. It is an explicitly documented coarse hypothesis when direct RoMa evidence is weak, not a neural success. --allow-inconsistent-routes permits a clearly labeled regional candidate when eligible route comparisons disagree; disagreement remains a failed check. Metadata candidate_region_manual_review and inconsistent_route_candidate_allowed record this, and reproduction requires the same explicit option. --allow-no-extra-route only permits a clearly labeled candidate if no extra structural route exists; it does not mark missing corroboration as passing.

Original final RGB is sampled once by global uniform scale, rotation and translation. T1 crop pixels match the original T1 exactly. Inference/descriptor preprocessing never changes saved RGB. Dense RoMa is not applied to final RGB. The largest horizontal geometric observation rectangle does not imply all pixels correspond reliably. Missing T3 in A017-1 is not synthesized; <=2 available periods are skipped. Source audit contains conditional point-CV, block-pose stability and route checks, not manual landmarks or independent ground-truth TRE. All weak edges remain labeled. Source RoMa normalized prediction caches remain in the execution results; they can be regenerated with collect_roma_region_candidates.py. Rebuilding inference does not replace an existing candidate cache; use a fresh results root for a fresh run.
