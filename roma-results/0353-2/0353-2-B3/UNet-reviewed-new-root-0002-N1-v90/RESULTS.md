# Explicit user-reviewed correction of one new T2 root

The user confirmed the long curved root shown as extension is entirely new in T2, and clarified that this instruction applies to THIS root, not all roots on the page. Its existing confirmed annotation identity is0002-N1/R20484, a new lateral first seen in T2. V89 represented that one root across11 model ownership IDs and wrongly inherited old state through a contact with R20038. In axis162, the original U-Net has26 persistent centerline pixels followed by489 new pixels. Only the first21 centerline points match the older T1 strand; that small contact prefix must not define the state of the lateral body. The split is at (2067,203), and the true older contact strand is retained.

This is an EXPLICITLY ANNOTATION-ASSISTED USER REVIEW CORRECTION, not additional neural training or a new label-blind model inference. The existing confirmed numbered-root ownership provides the scope, intersected with the immutable predicted foreground. Exactly3149 selected-root pixels become new;1049 were already new,1834 extension and266 persistent pixels are corrected, for2100 actual class changes. No rectangular region or entire page is repainted. Model ownership of the selected root fragments is unified as reviewed axis772. Other root pixels, manual labels, model weights, all binary root silhouettes and all T1/T3/T4 predictions remain bitwise unchanged. No gaps are filled and no root pixels are added or removed.

The reviewed connection/identity supervision is saved separately for future model training: fragments of this numbered lateral are same-root; its biological parent and the old strand it contacts are different root identities and must not transfer their established-state label. Original files and all V89 results are preserved. Future inference should retain this explicitly reviewed example rather than rerunning the earlier false-contact inheritance rule.

Same-reference scores are descriptive agreement after an annotation-assisted correction, NOT independent accuracy or evidence that unchanged network weights learned a new capability. Both original inner spatial audit and all valid annotated foreground scopes are retained; T1 and binary root misses are excluded from three-class temporal scores. Reference V87, including earlier T2 and T4 corrections, is unchanged.

| Scope | Variant | Persistent | New | Extension | Macro |
|---|---|---:|---:|---:|---:|
| original_inner_spatial_audit | V89_before_review | 0.976463 | 0.846642 | 0.340480 | 0.721195 |
| original_inner_spatial_audit | V90_after_explicit_user_review | 0.977363 | 0.889470 | 0.435415 | 0.767416 |
| all_valid_annotated_root_pixels | V89_before_review | 0.981722 | 0.892259 | 0.498443 | 0.790808 |
| all_valid_annotated_root_pixels | V90_after_explicit_user_review | 0.982058 | 0.907603 | 0.545661 | 0.811774 |

Downloads:

- [Fig1_Reported_new_root_before_after.tif](figures/Fig1_Reported_new_root_before_after.tif)
- [Fig2_Entire_numbered_new_root_before_after.tif](figures/Fig2_Entire_numbered_new_root_before_after.tif)
- [Fig3_CURRENT_ONLY_four_period_predictions_V90.tif](figures/Fig3_CURRENT_ONLY_four_period_predictions_V90.tif)
