# Saved continuation and monitoring results

All curves contain actual measurements. Epochs1–20 were replayed from archived starting weights; missing original validation losses were not invented. Epochs21 onward reload the old selected final weights, with a new optimizer. The boundary is marked on the figures.

| Model | Replay + additional epochs | Max relative training-loss difference | Max logged Dice difference | Original final retained | Selected epoch |
|---|---:|---:|---:|---|---:|
| U-Net | 20 + 10 | 0.125% | 0.000722 | True | 18 |
| DeepLabV3+ | 20 + 10 | 1.181% | 0.013253 | True | 12 |
| SegFormer-B0 | 20 + 10 | 0.224% | 0.001400 | True | 16 |

The proximity flags retain the advance thresholds: relative training-loss difference≤1% and logged validation-Dice difference≤0.005. Every model and epoch is retained in the comparison, including deviations; no unfavorable epochs are omitted. If either check fails, the primary loss figure keeps original epochs1–20training loss and leaves original validation loss missing. A separate supplementary TIFF displays the entire unmatched replay without treating it as reconstructed history.

One monitoring epoch comprises32batches×2augmented draws, following the old run; it is not a full sweep of the200training views. These200views are correlated crops of four original dates, not200independent acquisitions. All dates at a spatial coordinate share one allocation.

Training-selection monitoring uses the unchanged original validation labels. Final confusion matrices, ROC/PR and test scores use the reporting reference in0353-2-B3_final_paper_figures.zip. These are different reference definitions and their scores must not be combined or substituted. Same-position/teacher-influenced reference agreement is not an independent biological accuracy estimate. External independent annotated images:0.

Loss plots and LR plots are separate for each model, cover all completed epochs, and use no smoothing. All11TIFFs are RGB,600dpi,LZW-compressed. Original files and selected baselines remain untouched.
