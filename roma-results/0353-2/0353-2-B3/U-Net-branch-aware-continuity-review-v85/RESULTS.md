# All-field U-Net branch/shaft continuity review, no retraining

This continues the delivered V83 output, not a replacement U-Net baseline. All model weights stay fixed. The ConvLSTM/GAT continuation and root-identity features are reused at inference. No manual root IDs, source24 temporal classes or revised-reference classes enter this refinement. Source24 root IDs enter only the post-freeze audit.

Rules distinguish biological axes from mother/daughter families. A short confirmed old shaft may produce a much longer terminal extension; old/root length fraction alone cannot indicate new birth. A mother's established distal axis may keep or recover extension when earlier tip evidence or a same-axis stable extension anchor supports it. A separate new lateral remains new, including its attachment to either a persistent OR extending mother. Conversely, newly colored intervals with coherent earlier identity evidence become persistent. New-axis extension islands and short persistent islands are corrected conservatively. Shared crossing pixels stay in the incident-root types. All proposed intervals, junctions and unresolved cases are exported.

This is inference plus explicit topology/history decoding, not additional neural training or a claim of raw-model-only corrections. Ambiguous old/new axes and possible extension without reliable parent/tip evidence remain unchanged. Every path is processed computationally; this does not mean every biological root was independently verified by a human. No root pixel is added, removed, widened or shortened. T1 is unchanged.

The separate V84 reference revision changes exactly 457 T2 extension pixels to new at the user's explicit direction. That revision has no internal test/validation pixels. No other manual annotation is changed. Both reference versions are scored separately. Full-area reference agreement is in-sample descriptive agreement, not independent accuracy. The entire B3 is a development position. Conditional root-foreground temporal scores exclude binary segmentation misses and T1 baseline.

Validation evidence is weak: only two validation pixels changed, both corrected; extension validation Dice remains zero. Therefore these checks do not establish reliable extension generalization. Predictions are frozen before test scoring; no rule is fitted to test-score gains. Known user examples are development checks.

| Scope | Reference | Variant | New | Persistent | Extension | Macro |
|---|---|---|---:|---:|---:|---:|
| internal_fixed_root_spatial_holdout | source24_original | original_U_Net_V66 | 0.786193 | 0.968708 | 0.085708 | 0.613537 |
| internal_fixed_root_spatial_holdout | source24_original | previous_optimized_V83 | 0.826134 | 0.973565 | 0.126090 | 0.641930 |
| internal_fixed_root_spatial_holdout | source24_original | branch_aware_review_V85 | 0.830482 | 0.973197 | 0.179211 | 0.660963 |
| internal_fixed_root_spatial_holdout | source24_user_corrected | original_U_Net_V66 | 0.786193 | 0.968708 | 0.085708 | 0.613537 |
| internal_fixed_root_spatial_holdout | source24_user_corrected | previous_optimized_V83 | 0.826134 | 0.973565 | 0.126090 | 0.641930 |
| internal_fixed_root_spatial_holdout | source24_user_corrected | branch_aware_review_V85 | 0.830482 | 0.973197 | 0.179211 | 0.660963 |
| all_annotated_fixed_root_pixels | source24_original | original_U_Net_V66 | 0.836996 | 0.975512 | 0.288320 | 0.700276 |
| all_annotated_fixed_root_pixels | source24_original | previous_optimized_V83 | 0.853305 | 0.978570 | 0.265806 | 0.699227 |
| all_annotated_fixed_root_pixels | source24_original | branch_aware_review_V85 | 0.860605 | 0.978773 | 0.334624 | 0.724667 |
| all_annotated_fixed_root_pixels | source24_user_corrected | original_U_Net_V66 | 0.840323 | 0.975512 | 0.293428 | 0.703088 |
| all_annotated_fixed_root_pixels | source24_user_corrected | previous_optimized_V83 | 0.856611 | 0.978570 | 0.273064 | 0.702748 |
| all_annotated_fixed_root_pixels | source24_user_corrected | branch_aware_review_V85 | 0.863919 | 0.978773 | 0.343301 | 0.728664 |

Structural checks (different denominators from pixel Dice):

```json
{
  "root_records": 158,
  "root_conflicts": {
    "original_U_Net_V66": 24,
    "previous_optimized_V83": 11,
    "branch_aware_review_V85": 11
  },
  "junctions": 80,
  "third_type": {
    "original_U_Net_V66": {
      "junctions": 13,
      "pixels": 202
    },
    "previous_optimized_V83": {
      "junctions": 6,
      "pixels": 52
    },
    "branch_aware_review_V85": {
      "junctions": 5,
      "pixels": 51
    }
  }
}
```

Downloads and review:

- [Four-period comparison](figures/Fig1_Four_period_branch_aware_review.tif)
- [Mother extension and new lateral](figures/Fig2_Mother_extension_and_new_lateral.tif)
- [All 182 interval review entries: offline HTML](Root_by_root_review_offline.html)
- [Twenty corrected root intervals](Root_interval_proposals.csv)
- [All unresolved intervals](Review_cases_with_coordinates.csv)
- [Class Dice under two reference versions and two scopes](Dice_metrics_two_reference_versions.csv)

The viewer contains20 corrected-interval and162 unresolved-interval entries, with native-pixel zoom and same-ROI four-period switching. Junction-specific changes are listed separately in Branch_and_crossing_proposals.csv. Chromium document-load tests passed. The viewer displays prediction-axis audit IDs, not replacement manual root IDs.
