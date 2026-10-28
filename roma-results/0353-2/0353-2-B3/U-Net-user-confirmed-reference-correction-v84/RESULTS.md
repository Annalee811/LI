# User-confirmed T2 reference correction, no retraining

Only the pictured T2 extension interval (457 pixels; bbox [1423,105,1544,147]) is changed to newly observed. The user confirmed the annotation error. All old files, root contours, root IDs, weights and predictions are retained. No extension class outside this interval is changed.

The editable JSON uses temporal 1=new and 2=extension; derived class PNGs use 2=new and 3=extension. These code systems are deliberately kept distinct. The annotation revision is appended as metadata. No new root identifiers or structural labels are invented.

The corrected interval has no original validation/test pixels, so the previously reported internal test Dice is unchanged. Full annotated-area metrics are in-sample descriptive reference agreement, not independent accuracy or a training gain. All metrics are conditional on the frozen U-Net root foreground; T1 baseline is excluded from the three-class scores.

| Scope | Reference | Model | New | Persistent | Extension | Macro |
|---|---|---|---:|---:|---:|---:|
| internal_fixed_root_spatial_holdout | source24_before_correction | original_U_Net_V66 | 0.786193 | 0.968708 | 0.085708 | 0.613537 |
| internal_fixed_root_spatial_holdout | source24_before_correction | current_optimized_U_Net_V83 | 0.826134 | 0.973565 | 0.126090 | 0.641930 |
| internal_fixed_root_spatial_holdout | source24_user_corrected | original_U_Net_V66 | 0.786193 | 0.968708 | 0.085708 | 0.613537 |
| internal_fixed_root_spatial_holdout | source24_user_corrected | current_optimized_U_Net_V83 | 0.826134 | 0.973565 | 0.126090 | 0.641930 |
| all_annotated_fixed_root_pixels | source24_before_correction | original_U_Net_V66 | 0.836996 | 0.975512 | 0.288320 | 0.700276 |
| all_annotated_fixed_root_pixels | source24_before_correction | current_optimized_U_Net_V83 | 0.853305 | 0.978570 | 0.265806 | 0.699227 |
| all_annotated_fixed_root_pixels | source24_user_corrected | original_U_Net_V66 | 0.840323 | 0.975512 | 0.293428 | 0.703088 |
| all_annotated_fixed_root_pixels | source24_user_corrected | current_optimized_U_Net_V83 | 0.856611 | 0.978570 | 0.273064 | 0.702748 |
| user_confirmed_local_ROI | source24_before_correction | original_U_Net_V66 | 0.758834 | 0.912442 | 0.220859 | 0.630712 |
| user_confirmed_local_ROI | source24_before_correction | current_optimized_U_Net_V83 | 0.777536 | 0.926284 | 0.039634 | 0.581151 |
| user_confirmed_local_ROI | source24_user_corrected | original_U_Net_V66 | 0.923488 | 0.912442 | 0.391555 | 0.742495 |
| user_confirmed_local_ROI | source24_user_corrected | current_optimized_U_Net_V83 | 0.929348 | 0.926284 | 0.130653 | 0.662095 |
