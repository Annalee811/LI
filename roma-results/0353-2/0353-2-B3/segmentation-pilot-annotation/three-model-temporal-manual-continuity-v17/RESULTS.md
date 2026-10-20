# 0353-2-B3 — V14-based manual root-continuity refinement

V14 is preserved; V16 is not used. The exact original manual project is unchanged. Reference width/boundaries/extent follow the frozen U-Net binary roots; temporal classes at retained original typed pixels are unchanged. Added pixels without reliable original class evidence remain untyped.

## Downloads

- [Four-period integrated TIFF — 600 dpi, one A3 page, 72 panels](0353-2-B3_temporal_integrated_v17.tif)
- [Paper figures ZIP — includes V14/current crossing details](0353-2-B3_v17_paper_figures.zip)
- [Editable annotation, native RGB/reference overlays](0353-2-B3_v17_annotation_review.zip)
- [Matched before/after metrics](V14_matched_comparison.csv) · [Accuracy summary](accuracy_summary.csv) · [Original V14 published scores](V14_published_metrics.csv)
- Model results: [U-Net](0353-2-B3_v17_segmentation_unet.zip), [DeepLabV3+](0353-2-B3_v17_segmentation_deeplab.zip), [SegFormer](0353-2-B3_v17_segmentation_segformer.zip)
- Weights: [U-Net + SegFormer roots and all temporal heads](0353-2-B3_v17_weights_torch_and_temporal.zip), [TensorFlow DeepLabV3+ root model](0353-2-B3_v17_weights_deeplab.zip)
- [Complete scripts](0353-2-B3_v17_complete_scripts.zip)
- [Training tiles, exact original labels, morphology reference and RGB](0353-2-B3_v17_training_data.zip)

## Same-reference, same-window comparison

The following temporal macro metrics use identical internal test crops from saved full-field predictions and the manual-class/U-Net-morphology reference. They differ from the older V14 teacher-reference/tilewise protocol, so the quoted .7980/.4742/.5238 scores cannot be directly compared with this table.

| Model | Version | Precision | Recall | Dice | IoU |
|---|---|---:|---:|---:|---:|
|unet|V14|0.6688|0.6914|0.6788|0.5923|
|unet|manual_continuity_v17|0.6919|0.7056|0.6872|0.6139|
|deeplab|V14|0.4953|0.4964|0.4907|0.3716|
|deeplab|manual_continuity_v17|0.5079|0.5013|0.5042|0.3835|
|segformer|V14|0.4411|0.4646|0.4496|0.3433|
|segformer|manual_continuity_v17|0.5434|0.6136|0.5660|0.4210|

All three temporal heads were actually trained32epochs. Final validation selected the updated U-Net head; for DeepLab and SegFormer it retained V14 emissions combined with the new learned routing/decoder. This is recorded explicitly in decoder JSONs; new DeepLab/SegFormer fitting is not claimed superior. Binary networks remain frozen.

The same learned putative axes contain139/42/49NEW-and-GROWTH mixtures in V14 U-Net/DeepLab/SegFormer across the full upper T2–T4 fields; the final grammar forbids those mixtures. Zero is a constraint, **not independent biological root-identity accuracy**. Ambiguous crossings and growth fragments require review; review masks and axis records are provided.

Training/validation/internal test:200/20/24RGB views,150/15/18adjacent-date pairs. All dates at one window share a split; cross-split pixel overlap is zero. Historical same-field exposure and whole-axis context remain limitations. Independent external test images:0. Original-human audit and untyped-root coverage are supplied separately.

The integrated TIFF shows original RGB, morphology-adjusted manual-class reference and three final outputs for all4dates, with background, background removed and identical local zooms. Bluebaseline,purplepersistent,greennew,goldgrowth,grayuntypedreference. Native RGB is unchanged. Full fields include fitting pixels.

Paper ZIP:root-only views,temporal class Dice,root-axis inspection,S1actual learning curves,S2transition Dice,S3skeleton pixel-length agreement,S4window differences,S5errors,S6confusions,S7V14/current crossing details,S8axis consistency constraint counts. S7regions are chosen by V14 NEW/GROWTH content within existing internal test windows, not new-model scores. There are no independent manual physical-length measurements.

[Full method/reproduction](README.md) · [Plant Phenomics reporting notes](Plant_Phenomics_notes.md) · [Verification](verification.json) · [Checksums](SHA256SUMS.txt)
