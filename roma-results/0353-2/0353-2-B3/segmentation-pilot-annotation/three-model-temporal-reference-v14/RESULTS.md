# 0353-2-B3 — temporal reference v14 results

All three temporal heads completed24epochs/384updates. Root/background networks stayed frozen. Labels: T1baseline; later periods persistent,newly observed,tipgrowth relative to the immediately preceding period. Death excluded.

## Internal same-field results

Revised scores measure agreement with a U-Net-derived reference, not independent human accuracy. Original-reference scores are retained for audit; historical exposure also limits them.

| Model | Persistent Dice | New Dice | Growth Dice | Revised macroDice | Original macroDice |
|---|---:|---:|---:|---:|---:|
| U-Net | 0.9855 | 0.8859 | 0.5225 | 0.7980 | 0.4746 |
| DeepLabV3+ | 0.7755 | 0.6072 | 0.0399 | 0.4742 | 0.3614 |
| SegFormer-B0 | 0.7692 | 0.6039 | 0.1983 | 0.5238 | 0.3586 |

Compared with the inherited heads, revised-reference test macroDice changed by+0.0337,+0.0092,+0.0401. This does not imply the same improvement against original user labels: original-reference changes were−0.0025,+0.0020,−0.0409. Growth remains the difficult class.

Training/validation/internal-test paired examples:150/15/18, from50/5/6spatial windows with the four dates grouped. These include25shifted training views overlapping their own bases; zero cross-split pixel overlap. Independent external test images:0.

Selected epochs24/20/12; chain penalties1/8/2 selected using revised validation only. All18test pairs per model were reproduced by checkpoint reload; actual Chromium brush/undo/zoom/save/reimport checks passed. Original human JSON and source RGB/weights are preserved.

## Downloads

- [0353-2-B3_v14_annotation_review.zip](0353-2-B3_v14_annotation_review.zip) — 20.1MiB
- [0353-2-B3_v14_training_data.zip](0353-2-B3_v14_training_data.zip) — 67.7MiB
- [0353-2-B3_v14_segmentation_unet.zip](0353-2-B3_v14_segmentation_unet.zip) — 91.5MiB
- [0353-2-B3_v14_segmentation_deeplab.zip](0353-2-B3_v14_segmentation_deeplab.zip) — 92.3MiB
- [0353-2-B3_v14_segmentation_segformer.zip](0353-2-B3_v14_segmentation_segformer.zip) — 92.4MiB
- [0353-2-B3_v14_weights_torch_and_temporal.zip](0353-2-B3_v14_weights_torch_and_temporal.zip) — 68.5MiB
- [0353-2-B3_v14_weights_deeplab.zip](0353-2-B3_v14_weights_deeplab.zip) — 64.7MiB
- [0353-2-B3_v14_complete_scripts.zip](0353-2-B3_v14_complete_scripts.zip) — 0.2MiB
- [0353-2-B3_v14_paper_figures_RGB.zip](0353-2-B3_v14_paper_figures_RGB.zip) — 95.0MiB
- [0353-2-B3_v14_paper_figures_other.zip](0353-2-B3_v14_paper_figures_other.zip) — 14.8MiB

Extract all desired ZIPs into the same parent; they share0353-2-B3_temporal_v14. The annotation webpage embeds all four upper RGBs and references. Paper figures are split into two complete ZIPs to keep each file below GitHub’s size limit. Source references, full matrices, class metrics, length diagnostics, actual learning curves and SHA256s are retained. SeeREADME.md for exact commands and limitations.
