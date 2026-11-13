# V66: new local field, three contrasting classes and simplified legend

The figure has one shared Original RGB column, one shared Human reference column, and three prediction columns (U-Net, DeepLabV3+, SegFormer-B0) across T1–T4. Each model's loss curves and own class-wise evaluation sit directly above its prediction column. All curves share a 0–1.8 loss range; y-axis numbers are shown once. The corresponding metric rows and dates align across models. Original RGB and human reference are displayed once per date, not repeated in separate model groups. The twenty local panels use identical coordinates x=850:1270, y=650:970 (420×320 pixels) from 0353-2-B3. The larger local panel width is 39.5 mm.

The contrasting palette uses amber/orange #E69F35 (newly emerged), purple/magenta #B64E8B (elongating) and blue #3F75C4 (persistent). The common root-class row key identifies the colors in both score cells and RGB overlays. White denotes T1 baseline and gray denotes ignored reference pixels. Background RGB and label geometry remain unchanged. No new dilation, smoothing, annotations, root-instance assignments, registration, model training or inference are performed. Colors describe temporal classes rather than individual root identities. The single scale bar in the T1 RGB panel is 100 source pixels and applies to all local panels at identical magnification, without physical calibration.

Training/validation curves are the exact original 100-epoch logs of the three independent 25-channel V66 temporal heads with frozen binary branches. They are not end-to-end encoder retraining curves. Checkpoints at epochs 37, 14 and 63 were selected by maximum validation macro Dice on annotation set A, not by minimum validation loss. Curves are unsmoothed original measurements.

All 36 class-wise Precision/Recall/Dice/IoU scores and macro Dice use common frozen evaluation reference B over T2–T4 and 468,203 typed reference-root pixels. True background and ignored reference pixels are excluded, whereas missed roots remain false negatives. T1 is baseline and excluded from temporal scores. Macro Dice is 71.3237098784388%, 58.37331464872215% and 60.89011420429545%. These are within-development-position reference-agreement scores, not independent cross-tube validation or full-image binary segmentation accuracy. No new replicates, error bars or significance results are invented.

The human reference is the same frozen set B used for evaluation. It includes manual temporal assignments with pre-existing U-Net-assisted binary outline refinement; it is not an original pure unaided human stroke mask or independently blinded anatomical ground truth. Training set A and evaluation reference B are distinct. This scope is retained in the caption and source manifests.

White canvas; black actual Times New Roman regular/bold, 9.5–16 pt; 225×240 mm; 600-dpi JPG/PNG/LZW TIFF. At 190.5 mm reproduced width, nominal 9.5 pt text is approximately 8.0 pt. PDF text, curves and scale bars are vector and photographic image panels are raster. The source release is Annalee811/Temporal-segmentation-model at immutable commit 4615bcbd37442a3b11e5ad4533242df95f237fb6. All 26 source assets were hash-verified during data preparation; the current verifier also checks the original files when available.

The self-contained figure bundle includes exports, raw training CSVs and selection metadata, full-precision metrics/confusion matrices, compact full-label arrays for reproducible scoring, exact local RGB/label crops and overlays, plotting/verification scripts, bilingual caption, provenance and checksums. No full-field RGB photographs, model weights or raw font binaries are bundled. Historical project data and figures are unchanged.

```bash
python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

Supply licensed Times New Roman.ttf and Times New Roman Bold.ttf locally. Verification recomputes all scores, checks exact original loss curves/selection epochs, confirms that RGB and human reference occur only once per date, verifies model-column ownership and unchanged backgrounds/labels, and checks fonts, page bounds and output dimensions.

The legend is limited to Training/Validation line styles and the three class-row color keys; technical provenance, baseline and ignored-label explanations appear only in this caption/README. The new local ROI was chosen using reference morphology and class coverage only, without ranking model performance. All three models use the identical crop. Raw losses and all evaluation scores are unchanged from the preceding shared-reference figure.
