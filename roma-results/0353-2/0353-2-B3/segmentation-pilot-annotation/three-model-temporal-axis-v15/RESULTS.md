# 0353-2-B3 — human temporal labels and whole-axis routes, v15

The v14 U-Net temporal-teacher replacement has been withdrawn for this version. Every original typed temporal pixel is retained. U-Net supports only narrow morphology changes; original RGBs and full annotation JSON are preserved. Three paired temporal heads completed24epochs each.

## Inspect these first

- [RGB / root-axis routes / four-period temporal diagnostic — TIFF](Fig6_RGB_root_axis_continuity.tif)
- [PNG preview](Fig6_RGB_root_axis_continuity.png)
- [Editable human-first reference webpage package](0353-2-B3_v15_annotation_review.zip)

The root-axis figure uses local candidate IDs. Arrows indicate orientation, not proven growth polarity. Orange circles flag unresolved crossings. Whole-frame logits are blended before root-axis decisions. An accepted axis cannot mix NEW and GROWTH; old shafts can be PERSISTENT with terminal GROWTH. Outward previous-tip continuation fragments and new laterals at old shafts are treated separately.

## Internal development results

| Model | Persistent Dice | New Dice | Growth Dice | Width-adjusted human macroDice | Original narrow-reference macroDice |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.8663 | 0.7091 | 0.1163 | 0.5639 | 0.4335 |
| DeepLabV3+ | 0.7169 | 0.5470 | 0.1514 | 0.4718 | 0.3774 |
| SegFormer-B0 | 0.6966 | 0.5025 | 0.0050 | 0.4014 | 0.3223 |

Growth remains weak; these outputs are not established biological root identities or publication-level temporal accuracy. Raw-pixel macroDice before whole-axis rules was0.6391/0.4490/0.4390against the width-adjusted human reference. Rules improve anatomic consistency but do not improve all pixel metrics. See metrics_overview.csv and root_axis_continuity.csv. Do not compare this target numerically with v14 U-Net-temporal-teacher agreement as if the references were unchanged.

Frozen root DeepLab uses TensorFlow1.12;the U-Net and SegFormer root stages and all3temporal heads usePyTorch. Framework choice does not resolve roots at crossings. All3label-free checkpoint reloads reproduced all4period maps pixel for pixel. Original typed classes changed0.

Training/validation/internal-test pairs150/15/18, from200/20/24RGBviews;all dates at one location stay together. Shifted training views overlap their bases, and full-axis context crosses crop boundaries. Historical same-field exposure and iterative development prevent a new blind-test claim. Independent external test images0.

## Downloads

- [0353-2-B3_v15_annotation_review.zip](0353-2-B3_v15_annotation_review.zip) — 19.6MiB
- [0353-2-B3_v15_training_data.zip](0353-2-B3_v15_training_data.zip) — 64.6MiB
- [0353-2-B3_v15_segmentation_unet.zip](0353-2-B3_v15_segmentation_unet.zip) — 84.3MiB
- [0353-2-B3_v15_segmentation_deeplab.zip](0353-2-B3_v15_segmentation_deeplab.zip) — 85.7MiB
- [0353-2-B3_v15_segmentation_segformer.zip](0353-2-B3_v15_segmentation_segformer.zip) — 85.2MiB
- [0353-2-B3_v15_weights_torch_and_temporal.zip](0353-2-B3_v15_weights_torch_and_temporal.zip) — 60.7MiB
- [0353-2-B3_v15_weights_deeplab.zip](0353-2-B3_v15_weights_deeplab.zip) — 64.7MiB
- [0353-2-B3_v15_complete_scripts.zip](0353-2-B3_v15_complete_scripts.zip) — 0.3MiB
- [0353-2-B3_v15_paper_RGB_full.zip](0353-2-B3_v15_paper_RGB_full.zip) — 36.4MiB
- [0353-2-B3_v15_paper_RGB_local.zip](0353-2-B3_v15_paper_RGB_local.zip) — 58.7MiB
- [0353-2-B3_v15_paper_axes_and_evaluation.zip](0353-2-B3_v15_paper_axes_and_evaluation.zip) — 79.9MiB

All ZIPs share0353-2-B3_temporal_v15. Extract into the same parent. Original/revisedreferences,sourceRGBs,pairedtiles,weights,fullscripts,axis/reviewmasks,all4periodoutputs,tables,and600dpiTIFFfigures are included. No other position is predicted. Original labels are retained; model predictions are not relabeled as new manual truth.
