# Original U-Net vs V89 comparison

The original baseline is V66 original U-Net (100 completed training epochs, selected epoch37), checkpoint SHA25658a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382. This is the original prediction corresponding to historical reference20 new-root Dice0.8229. That historical score is not reused as a score under a different reference. Intermediate V85 or V88 predictions are NOT substituted for the original. The right-hand prediction is the exact published V89 output.

Figures show original RGB, ORIGINAL U-Net (V66), OPTIMIZED output (V89), clearly labeled. Fig1 has overlay and white-background roots for all four dates. Fig2 uses the same matched native ROI in all four periods. Fig3 compares the three reported T2 regions, using identical native crops. All root contours, widths/lengths and original prediction files remain unchanged. T1 is identical. This export performs no training, inference, postprocessing or annotation modification.

V89 is the inherited learned ConvLSTM/GAT features plus structural axis/junction decoding result; it is not a newly retrained U-Net. Its low-confidence state decisions and anatomical ownership limitations remain as reported in V89. Consistency-rule compliance is not independent accuracy. Optional metrics are read from already verified results under the SAME V87 reference, SAME fixed root foreground and SAME spatial scopes, with equal class supports and evaluated pixel counts asserted. T1 is excluded from three-class metrics. The original inner spatial audit is a development region and full-area agreement is in-sample descriptive agreement. No binary root-miss penalty is added.

| Scope | Variant | Persistent | New | Extension | Macro |
|---|---|---:|---:|---:|---:|
| original_inner_spatial_audit | original_U_Net_V66 | 0.968708 | 0.818142 | 0.099134 | 0.628661 |
| original_inner_spatial_audit | optimized_V89 | 0.976463 | 0.846642 | 0.340480 | 0.721195 |
| all_valid_annotated_root_pixels | original_U_Net_V66 | 0.975512 | 0.851788 | 0.312411 | 0.713237 |
| all_valid_annotated_root_pixels | optimized_V89 | 0.981722 | 0.892259 | 0.498443 | 0.790808 |

Downloads:

- [Fig1_Original_UNet_vs_V89_four_period_comparison.tif](figures/Fig1_Original_UNet_vs_V89_four_period_comparison.tif)
- [Fig2_Original_UNet_vs_V89_matched_four_period_detail.tif](figures/Fig2_Original_UNet_vs_V89_matched_four_period_detail.tif)
- [Fig3_Original_UNet_vs_V89_reported_T2_details.tif](figures/Fig3_Original_UNet_vs_V89_reported_T2_details.tif)

- [Same-reference metrics CSV](Original_UNet_vs_V89_same_reference_metrics.csv)
