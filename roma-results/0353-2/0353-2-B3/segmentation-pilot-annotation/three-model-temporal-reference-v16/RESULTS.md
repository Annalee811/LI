# 0353-2-B3 — temporal reference refinement v16

All three temporal heads were trained for 24 epochs. Original human labels remain preserved. The editable revised reference comes from the latest v15 U-Net whole-axis outputs; it includes both morphology and temporal classes. Binary root detectors remain frozen.

## Downloads

- [Integrated four-date TIFF — A3 landscape, 600 dpi, 72 panels](0353-2-B3_temporal_integrated_v16.tif)
- [Supplemental paper figures](0353-2-B3_v16_paper_figures.zip)
- [Accuracy table](accuracy_summary.csv)
- [Editable four-date annotation and native reference RGB/overlays](0353-2-B3_v16_annotation_review.zip)
- Results: [U-Net](0353-2-B3_v16_segmentation_unet.zip), [DeepLabV3+](0353-2-B3_v16_segmentation_deeplab.zip), [SegFormer-B0](0353-2-B3_v16_segmentation_segformer.zip)
- Weights: [Torch root models + all temporal heads](0353-2-B3_v16_weights_torch_and_temporal.zip), [TensorFlow DeepLab root model](0353-2-B3_v16_weights_deeplab.zip)
- [Complete scripts](0353-2-B3_v16_complete_scripts.zip)
- [Training tiles, original human project, revised reference and RGB](0353-2-B3_v16_training_data.zip)

## Measured internal results

| Reference | Model | Precision | Recall | Macro Dice | IoU | Selected epoch |
|---|---|---:|---:|---:|---:|---:|
| U-Net-derived reference | U-Net | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 / 24 |
| Original human audit | U-Net | 0.3794 | 0.5175 | 0.4335 | 0.3084 | 0 / 24 |
| U-Net-derived reference | DeepLabV3+ | 0.4776 | 0.4505 | 0.4604 | 0.3293 | 0 / 24 |
| Original human audit | DeepLabV3+ | 0.3450 | 0.4402 | 0.3774 | 0.2507 | 0 / 24 |
| U-Net-derived reference | SegFormer-B0 | 0.4139 | 0.4291 | 0.4164 | 0.3077 | 4 / 24 |
| Original human audit | SegFormer-B0 | 0.2879 | 0.4121 | 0.3259 | 0.2185 | 4 / 24 |

These are three-class temporal macro metrics over the 18 internal test date-pairs, excluding background and T1 baseline. Training/validation/internal test comprise 200/20/24 RGB views or 150/15/18 adjacent-date pairs. All dates at one spatial window share a partition. External independent test images: **0**.

**U-Net agreement with its own generated reference is circular. A score of 1.0 is not independently measured biological accuracy.** Original-human agreement is a separate same-field audit. Earlier weights have same-field exposure; full-axis context can cross split windows. U-Net and DeepLab selected their starting temporal checkpoints after 24 real training epochs failed to improve validation whole-axis Dice; SegFormer selected epoch 4. Last-epoch weights and all histories are retained.

The integrated TIFF has four date blocks. Each shows original RGB controls, revised reference and three model outputs, full/local overlays and background-removed temporal classes. Blue: T1 baseline; purple: persistent; green: newly observed; gold: tip growth. Native RGB is unchanged. The same 256×256 native-pixel ROI is shown across dates. Full fields include training pixels and are qualitative.

The paper archive includes S1 actual learning curves, S2 transition Dice, S3 skeleton-length agreement, S4 per-window heterogeneity, S5 class errors (green correct, pink false positive, orange false negative), S6 confusion matrices, plus root-only views, class Dice and root-axis inspection. Skeleton pixel counts are not independently measured physical root lengths.

[Protocol and reproduction](README.md) · [Plant Phenomics reporting notes](Plant_Phenomics_notes.md) · [Validation record](verification.json) · [SHA256 checksums](SHA256SUMS.txt)
