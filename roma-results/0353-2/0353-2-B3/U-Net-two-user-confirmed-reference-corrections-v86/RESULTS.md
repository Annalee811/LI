# Two user-confirmed reference corrections; fixed U-Net predictions

No fitting, retraining, model-weight update, postprocessing update or prediction edit is performed in V86. The original V66, optimized V83 and reviewed V85 prediction maps are each scored unchanged under all reference versions. Their previous quantitative records remain preserved.

Reference revision1: T2 bbox [1423,105,1544,147],457 extension pixels changed to new, as explicitly confirmed by the user. Revision2: the pictured diagonal new lateral0017-N6 (existing manual root R20569), bbox [1227,458,1392,653],916 extension pixels changed to new. This includes827 pixels on the diagonal and89 on the same-root short attachment-side interval. No mother-root or other root class is changed. Root IDs and all four root masks are unchanged. The cumulative reference change is1373 T2 pixels; all other dates/layers and the original24 JSON are preserved. JSON temporal codes1=new,2=extension differ from derived PNG codes2=new,3=extension.

These are manual reference corrections prompted by user review, not improvements from training. Scores against the corrected reference are internal reference agreement and cannot be interpreted as newly demonstrated independent accuracy. All scopes are conditional on the immutable root foreground; segmentation omissions are outside this temporal-only task. T1 baseline is excluded from three-class metrics. Both full-area and the unchanged internal-test mask are exported separately.

| Scope | Reference | Prediction | New | Persistent | Extension | Macro |
|---|---|---|---:|---:|---:|---:|
| internal_fixed_root_spatial_holdout | source24_original | original_U_Net_V66 | 0.786193 | 0.968708 | 0.085708 | 0.613537 |
| internal_fixed_root_spatial_holdout | source24_original | optimized_U_Net_V83 | 0.826134 | 0.973565 | 0.126090 | 0.641930 |
| internal_fixed_root_spatial_holdout | source24_original | reviewed_U_Net_V85 | 0.830482 | 0.973197 | 0.179211 | 0.660963 |
| internal_fixed_root_spatial_holdout | first_user_correction_V84 | original_U_Net_V66 | 0.786193 | 0.968708 | 0.085708 | 0.613537 |
| internal_fixed_root_spatial_holdout | first_user_correction_V84 | optimized_U_Net_V83 | 0.826134 | 0.973565 | 0.126090 | 0.641930 |
| internal_fixed_root_spatial_holdout | first_user_correction_V84 | reviewed_U_Net_V85 | 0.830482 | 0.973197 | 0.179211 | 0.660963 |
| internal_fixed_root_spatial_holdout | both_user_corrections_V86 | original_U_Net_V66 | 0.803997 | 0.968708 | 0.092026 | 0.621577 |
| internal_fixed_root_spatial_holdout | both_user_corrections_V86 | optimized_U_Net_V83 | 0.842675 | 0.973565 | 0.143300 | 0.653180 |
| internal_fixed_root_spatial_holdout | both_user_corrections_V86 | reviewed_U_Net_V85 | 0.847046 | 0.973197 | 0.203822 | 0.674688 |
| all_annotated_fixed_root_pixels | source24_original | original_U_Net_V66 | 0.836996 | 0.975512 | 0.288320 | 0.700276 |
| all_annotated_fixed_root_pixels | source24_original | optimized_U_Net_V83 | 0.853305 | 0.978570 | 0.265806 | 0.699227 |
| all_annotated_fixed_root_pixels | source24_original | reviewed_U_Net_V85 | 0.860605 | 0.978773 | 0.334624 | 0.724667 |
| all_annotated_fixed_root_pixels | first_user_correction_V84 | original_U_Net_V66 | 0.840323 | 0.975512 | 0.293428 | 0.703088 |
| all_annotated_fixed_root_pixels | first_user_correction_V84 | optimized_U_Net_V83 | 0.856611 | 0.978570 | 0.273064 | 0.702748 |
| all_annotated_fixed_root_pixels | first_user_correction_V84 | reviewed_U_Net_V85 | 0.863919 | 0.978773 | 0.343301 | 0.728664 |
| all_annotated_fixed_root_pixels | both_user_corrections_V86 | original_U_Net_V66 | 0.846387 | 0.975512 | 0.301540 | 0.707813 |
| all_annotated_fixed_root_pixels | both_user_corrections_V86 | optimized_U_Net_V83 | 0.863006 | 0.978570 | 0.288496 | 0.710024 |
| all_annotated_fixed_root_pixels | both_user_corrections_V86 | reviewed_U_Net_V85 | 0.870325 | 0.978773 | 0.361765 | 0.736954 |

Second correction split counts: {"unassigned": 86, "train": 430, "validation": 0, "internal_test": 400}
