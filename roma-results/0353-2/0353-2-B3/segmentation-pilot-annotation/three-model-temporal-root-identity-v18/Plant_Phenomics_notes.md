# Reporting notes — V18 temporal root continuity

The exact original human project is retained. Root width, boundaries and root-mask coverage follow the frozen U-Net morphology reference used in V17. V18 selectively changes 3,581 temporal pixels (T1: 0; T2: 2,142; T3: 885; T4: 554) in 120 logged root events, using the delivered V17 U-Net predictions, previous manual traces and candidate root geometry. These model-supported edits remain pending expert review; they are not independent manual truth. Unknown pixels, native RGB and binary root masks are unchanged. Death is not classified.

Three temporal pipelines were optimized in two stages: 32 epochs, followed by 16 epochs from the selected first-stage checkpoints. Binary U-Net, TensorFlow DeepLabV3+ and SegFormer-B0 networks remain frozen. The paired temporal heads share a 31-channel U-shaped design implemented in PyTorch. Do not describe this as end-to-end retraining of three temporal backbone architectures. Actual first-stage and final histories, selected and last weights, starting weights, teacher evidence and source snapshots are provided.

Final checkpoint and calibration selection uses only the unchanged V17 validation reference. All three final emission blends select weight zero for the newly optimized final head, retaining the first-stage selected emission head with the final learned junction router and root-level decoder. The additional 16 training epochs therefore did not improve the validation-selected emission head. This fact must be reported alongside the executed training, rather than implying that the last head is the best one.

The primary comparison uses identical 256-pixel windows and the unchanged V17 reference. Test temporal macro Dice changes from 0.6872 to 0.6585 (U-Net), 0.5042 to 0.6021 (DeepLabV3+) and 0.5660 to 0.5733 (SegFormer-B0). Corrected-reference Dice values are 0.6654, 0.6049 and 0.5785, respectively; these are model-supported agreement measures. Precision, Recall and IoU, original-human audit results, class-specific results and matched before/after scores are separate CSV tables. Do not substitute the corrected-reference score for independent accuracy or compare it directly with historical V14 teacher-reference values.

Candidate roots are traced before image tiling. Multi-scale tangent, width, RGB appearance, lateral entry/exit position and previous predicted-root correspondence inform routing at crossings. Root decoding allows an entirely newly observed root, or a persistent shaft with terminal extension. A new lateral is not reclassified as extension merely because it touches an old shaft. The absence of NEW/PERSISTENT or NEW/GROWTH mixing on accepted candidate roots is an enforced constraint, not independently measured biological identity accuracy. Ambiguous crossings are flagged. Displayed candidate IDs and cross-date associations are provisional; an unmatched root is not automatically dead. Shared raster crossing pixels have a single owner; single-root plots additionally show each candidate's complete path and class runs.

Training/validation/internal-test RGB window counts are 200/20/24, equivalent to 150/15/18 adjacent-date pairs. Dates at the same window share the same split. Cross-split pixel overlap is zero, but whole-field geometry crosses window boundaries, historical checkpoints have same-field exposure, and reported problem regions were used for development. These are correlated same-position development results, not a fresh blind external test. Independent external test images: 0. Skeleton length regression uses mask-derived pixel counts, not independently measured physical root lengths.

## Suggested figure placement

- Main integrated TIFF: four periods, original RGB, selectively corrected reference and all three outputs; full-field and local overlays plus background-removed roots. One A3 landscape page, 600 dpi. Full-field panels include training pixels and are descriptive.
- S1: actual final 16-epoch objective and validation trajectories. First-stage 32-epoch raw histories are also supplied. No synthetic smoothing or invented epochs.
- S2: transition-specific temporal Dice against the frozen reference.
- S3: skeleton pixel-count agreement for all three models, clearly identifying the original and frozen reference. This is not independent physical root-length validation.
- S4: all 18 internal test pairs and their between-window variation; repeated dates at six spatial windows are correlated.
- S5: class error localization; green correct class, pink false positive, orange false negative.
- S6: temporal class confusion matrices against the frozen reference.
- S7: the two user-reported crossing details, with T1/T2 RGB, V17 U-Net, corrected reference and three final outputs. Regions were fixed from the reported problems, not chosen by current scores.
- S8: one candidate root highlighted per panel, with per-root class runs. Arrows indicate orientation, not verified growth polarity.

The main text should distinguish geometric continuity, temporal class agreement and independently validated biological identity. This experiment supports inspectable continuity-constrained predictions, with a mixed accuracy outcome; it does not establish publication-level generalization to other tubes. Retain V17 for comparison, especially because U-Net primary Dice declined.

## Code and data availability

Use the immutable GitHub commit URL in the new RESULTS.md index. The editor ZIP opens locally in Chrome/Edge and supports reviewing the proposal. Original human annotations, frozen V17 reference, final revised project and root-by-root evidence are separately preserved. State the exact reference used in each table and figure. An archival repository release with a DOI can be added later if deposited; no DOI has been assigned here. These notes do not assert current journal-specific upload requirements. No Word file is needed.

## Methodological references

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://doi.org/10.1007/978-3-319-24574-4_28
- Chen LC et al. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV 2018. https://doi.org/10.1007/978-3-030-01234-2_49
- Xie E et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021. https://arxiv.org/abs/2105.15203
- Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network. 2015. https://arxiv.org/abs/1503.02531
- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet 1986. https://doi.org/10.1016/S0140-6736(86)90837-8
