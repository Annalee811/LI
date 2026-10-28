# T4 reference-only change: extension to new

The user explicitly instructed that ALL T4 manual extension labels become new. Exactly784 T4 temporal pixels are reassigned; no extension label remains in the T4 manual temporal layer. The earlier1373 explicitly confirmed T2 corrections remain, for2157 cumulative changed pixels relative to original24. All root masks, widths, lengths, identifiers, other annotation layers and prior files are preserved.

No model is fitted or retrained and no prediction is edited. Original V66, optimized V83 and reviewed V85 predictions are all rescored unchanged. Score changes between V86 and V87 come only from the reference revision; they are not new training gains. The original24 reference and prior quantitative results remain available. Conditional temporal Dice on the fixed root foreground is reported; binary segmentation omissions are outside this task and T1 baseline is excluded. Full annotated-area scores are descriptive in-sample agreement. Internal spatial masks belong to the same development position and are not an independent tube test.

T4 has zero reference extension support after this revision. Its extension-class entry is0 when a model still predicts extension (false positives), or undefined when absent in both reference and prediction. For T4-only summaries, the macro over reference-present classes uses persistent and new only; the combined T2/T3/T4 score retains all three classes because extension remains in earlier dates. CSV exports both reference support and predicted extension counts.

| Scope | Reference | Periods | Prediction | Persistent | New | Extension | Three-class macro |
|---|---|---|---|---:|---:|---:|---:|
| internal_fixed_root_spatial_holdout | source24_original | T2_T3_T4_combined | original_U_Net_V66 | 0.968708 | 0.786193 | 0.085708 | 0.613537 |
| internal_fixed_root_spatial_holdout | source24_original | T2_T3_T4_combined | optimized_U_Net_V83 | 0.973565 | 0.826134 | 0.126090 | 0.641930 |
| internal_fixed_root_spatial_holdout | source24_original | T2_T3_T4_combined | reviewed_U_Net_V85 | 0.973197 | 0.830482 | 0.179211 | 0.660963 |
| internal_fixed_root_spatial_holdout | source24_original | T4_only | original_U_Net_V66 | 0.994482 | 0.651054 | 0.000000 | 0.548512 |
| internal_fixed_root_spatial_holdout | source24_original | T4_only | optimized_U_Net_V83 | 0.995450 | 0.653396 | 0.000000 | 0.549615 |
| internal_fixed_root_spatial_holdout | source24_original | T4_only | reviewed_U_Net_V85 | 0.994981 | 0.663102 | 0.000000 | 0.552694 |
| internal_fixed_root_spatial_holdout | two_T2_corrections_V86 | T2_T3_T4_combined | original_U_Net_V66 | 0.968708 | 0.803997 | 0.092026 | 0.621577 |
| internal_fixed_root_spatial_holdout | two_T2_corrections_V86 | T2_T3_T4_combined | optimized_U_Net_V83 | 0.973565 | 0.842675 | 0.143300 | 0.653180 |
| internal_fixed_root_spatial_holdout | two_T2_corrections_V86 | T2_T3_T4_combined | reviewed_U_Net_V85 | 0.973197 | 0.847046 | 0.203822 | 0.674688 |
| internal_fixed_root_spatial_holdout | two_T2_corrections_V86 | T4_only | original_U_Net_V66 | 0.994482 | 0.651054 | 0.000000 | 0.548512 |
| internal_fixed_root_spatial_holdout | two_T2_corrections_V86 | T4_only | optimized_U_Net_V83 | 0.995450 | 0.653396 | 0.000000 | 0.549615 |
| internal_fixed_root_spatial_holdout | two_T2_corrections_V86 | T4_only | reviewed_U_Net_V85 | 0.994981 | 0.663102 | 0.000000 | 0.552694 |
| internal_fixed_root_spatial_holdout | two_T2_and_all_T4_corrections_V87 | T2_T3_T4_combined | original_U_Net_V66 | 0.968708 | 0.818142 | 0.099134 | 0.628661 |
| internal_fixed_root_spatial_holdout | two_T2_and_all_T4_corrections_V87 | T2_T3_T4_combined | optimized_U_Net_V83 | 0.973565 | 0.855837 | 0.163644 | 0.664349 |
| internal_fixed_root_spatial_holdout | two_T2_and_all_T4_corrections_V87 | T2_T3_T4_combined | reviewed_U_Net_V85 | 0.973197 | 0.858445 | 0.231687 | 0.687776 |
| internal_fixed_root_spatial_holdout | two_T2_and_all_T4_corrections_V87 | T4_only | original_U_Net_V66 | 0.994482 | 0.864465 | 0.000000 | 0.619649 |
| internal_fixed_root_spatial_holdout | two_T2_and_all_T4_corrections_V87 | T4_only | optimized_U_Net_V83 | 0.995450 | 0.866437 | 0.000000 | 0.620629 |
| internal_fixed_root_spatial_holdout | two_T2_and_all_T4_corrections_V87 | T4_only | reviewed_U_Net_V85 | 0.994981 | 0.852295 | 0.000000 | 0.615759 |
| all_annotated_fixed_root_pixels | source24_original | T2_T3_T4_combined | original_U_Net_V66 | 0.975512 | 0.836996 | 0.288320 | 0.700276 |
| all_annotated_fixed_root_pixels | source24_original | T2_T3_T4_combined | optimized_U_Net_V83 | 0.978570 | 0.853305 | 0.265806 | 0.699227 |
| all_annotated_fixed_root_pixels | source24_original | T2_T3_T4_combined | reviewed_U_Net_V85 | 0.978773 | 0.860605 | 0.334624 | 0.724667 |
| all_annotated_fixed_root_pixels | source24_original | T4_only | original_U_Net_V66 | 0.994585 | 0.718149 | 0.000000 | 0.570911 |
| all_annotated_fixed_root_pixels | source24_original | T4_only | optimized_U_Net_V83 | 0.995022 | 0.726313 | 0.000000 | 0.573778 |
| all_annotated_fixed_root_pixels | source24_original | T4_only | reviewed_U_Net_V85 | 0.995163 | 0.750539 | 0.000000 | 0.581901 |
| all_annotated_fixed_root_pixels | two_T2_corrections_V86 | T2_T3_T4_combined | original_U_Net_V66 | 0.975512 | 0.846387 | 0.301540 | 0.707813 |
| all_annotated_fixed_root_pixels | two_T2_corrections_V86 | T2_T3_T4_combined | optimized_U_Net_V83 | 0.978570 | 0.863006 | 0.288496 | 0.710024 |
| all_annotated_fixed_root_pixels | two_T2_corrections_V86 | T2_T3_T4_combined | reviewed_U_Net_V85 | 0.978773 | 0.870325 | 0.361765 | 0.736954 |
| all_annotated_fixed_root_pixels | two_T2_corrections_V86 | T4_only | original_U_Net_V66 | 0.994585 | 0.718149 | 0.000000 | 0.570911 |
| all_annotated_fixed_root_pixels | two_T2_corrections_V86 | T4_only | optimized_U_Net_V83 | 0.995022 | 0.726313 | 0.000000 | 0.573778 |
| all_annotated_fixed_root_pixels | two_T2_corrections_V86 | T4_only | reviewed_U_Net_V85 | 0.995163 | 0.750539 | 0.000000 | 0.581901 |
| all_annotated_fixed_root_pixels | two_T2_and_all_T4_corrections_V87 | T2_T3_T4_combined | original_U_Net_V66 | 0.975512 | 0.851788 | 0.312411 | 0.713237 |
| all_annotated_fixed_root_pixels | two_T2_and_all_T4_corrections_V87 | T2_T3_T4_combined | optimized_U_Net_V83 | 0.978570 | 0.868258 | 0.303538 | 0.716789 |
| all_annotated_fixed_root_pixels | two_T2_and_all_T4_corrections_V87 | T2_T3_T4_combined | reviewed_U_Net_V85 | 0.978773 | 0.874175 | 0.379577 | 0.744175 |
| all_annotated_fixed_root_pixels | two_T2_and_all_T4_corrections_V87 | T4_only | original_U_Net_V66 | 0.994585 | 0.824105 | 0.000000 | 0.606230 |
| all_annotated_fixed_root_pixels | two_T2_and_all_T4_corrections_V87 | T4_only | optimized_U_Net_V83 | 0.995022 | 0.832512 | 0.000000 | 0.609178 |
| all_annotated_fixed_root_pixels | two_T2_and_all_T4_corrections_V87 | T4_only | reviewed_U_Net_V85 | 0.995163 | 0.831930 | 0.000000 | 0.609031 |

T4 changed pixels by spatial mask: {"unassigned": 82, "train": 381, "validation": 0, "internal_test": 321}
