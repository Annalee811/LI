# 0353-2-B3 — version 16 temporal-reference refinement

This version was requested to replace the editable temporal reference with the latest v15 U-Net structured outputs, including root morphology, baseline, newly observed roots, persistent segments and tip growth. Original complete human JSON and original class rasters remain separate and unmodified. The v15 human-first reference is retained as a separate file. **The revised reference is model-derived; agreement with it is not independent manual accuracy.**

Only the registered upper 2541 × 1108 native RGB field is used. Original RGB colors and pixels are preserved. T1 roots are baseline only. T2–T4 are each compared with their immediately preceding date. Death is excluded. An axis may be new, or persistent with terminal growth; an outward continuation at a real previous tip may be entirely growth. Root-axis geometry at crossings is putative, not independently verified root identity.

## Training and evaluation

The same 25 base training, 5 validation and 6 internal test spatial windows are used for all models and dates. 25 shifted training views add variety without overlapping validation/test pixels. There are 200/20/24 RGB views, corresponding to 150/15/18 adjacent-date pairs; augmentations are not counted as new independent images. All four dates at one location share a split. No easy-tile filtering or teacher-confidence selection is applied. This is a same-position development experiment with historical same-field exposure and full-axis context across window boundaries. External independent test images: **0**.

The previously trained U-Net, DeepLabV3+ and SegFormer-B0 binary root detectors are **frozen**. Their own root masks/probabilities remain different; student models do not borrow the U-Net gate. The three corresponding 31-channel paired temporal heads are actually fine-tuned for 24 epochs, 384 optimizer updates each, from their own v15 checkpoints. Root RGB/shape, previous/current probabilities and whole-axis correspondence form the features. The temporal heads are PyTorch; the original DeepLabV3+ binary detector is TensorFlow 1.12. This is a two-stage pipeline, not three newly end-to-end-trained temporal architectures.

Optimizer: AdamW, learning rate 1e-4, weight decay 1e-4, cosine decay to 1e-5. Batch size 4; 64 paired draws per epoch; identical draw/augmentation sequence across models. Class weights are inverse-square-root frequency ratios capped at 8. Loss combines weighted cross entropy, class Dice, boundary BCE, same-label neighborhood and same-class skeleton continuity. CUDA is detected automatically; this run uses CPU BF16 inference/FP32 objective.

Checkpoints are selected only by full-frame, whole-axis **validation** macro Dice against the U-Net-derived reference, every four epochs, including epoch 0. The inherited v15 decoder strength is fixed during checkpoint selection, then the fixed grid [0, 0.5, 1, 2] is selected on the same validation windows. Test windows and original human labels do not select the checkpoint or decoder. Original-label evaluation is a separate development audit. If fine-tuning fails to beat the starting checkpoint, the latter is retained; the training history and last weights are still saved. In particular, a U-Net checkpoint identical to the reference generator can have perfect teacher agreement by construction. This is not evidence of perfect biological accuracy.

Temporal macro metrics pool confusion counts over the 18 internal test date-pairs and average the three temporal classes; background and the T1 baseline are excluded from the macro. End-to-end metrics retain upstream root misses/false positives. Per-class, per-transition, per-pair and conditional metrics are also saved. Root-length comparisons use skeleton pixel counts, not calibrated physical length or independent manual measurements.

## Reproduction

Use the repository layout preserved in the ZIPs. Install requirements in a fresh virtual environment; PyTorch automatically detects CUDA. Source dependencies `temporal_v13`, `dataset_identity.py`, binary training sources, TensorFlow Docker runtime and uploaded DeepLab source are in the scripts package. Prior v15 weights are included in the weights package, alongside the selected and last v16 weights.

```bash
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_reference_v16/prepare.py --run results/NEW_v16 --roots results/B3_annotation_refinement_v1 --previous results/B3_temporal_axis_v15
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_reference_v16/cache.py --run results/NEW_v16 --roots results/B3_annotation_refinement_v1 --previous results/B3_temporal_axis_v15
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_reference_v16/run_all.py --run results/NEW_v16 --roots results/B3_annotation_refinement_v1 --previous results/B3_temporal_axis_v15
```

`prepare.py` reuses existing prediction-only caches if available. If caches are not distributed, regenerate them with `cache.py`; no manual labels enter those features. For label-free prediction on a new aligned four-period position, invoke `predict_RGB.py --head ... --decoder ... --rgb-dir <canonical-position-directory> --root-dir <frozen-root-outputs> --output <new-directory>`. `reload_all.py` independently recomputes all four predictions and checks pixel-exact reproducibility against saved results without reading manual labels.

Open `annotation_review/annotate.html` in Chrome/Edge to edit the embedded four-date reference, with Ctrl+Z, a fixed toolbar, up to 60× zoom and unchanged zoom across dates. JSON saves and exports retain version/source information. Do not overwrite the original human project.

## Figures

The headline RGB TIFF is one A3 landscape page at 600 dpi. Four date blocks each show the original full/local RGB, revised reference and all three outputs, as full overlays, full background-removed temporal classes, local overlays and matching local background-removed classes. Original RGB root cutouts with alpha are also retained in each model result package. Native input resolution is unchanged; 600-dpi layout scaling does not create new biological detail.

Blue = baseline, purple = persistent, green = newly observed, gold = tip growth. Full-field panels include training pixels and are qualitative. Supplemental figures use measured histories and held-out spatial windows; no fabricated values or curve smoothing are used. Model-derived reference provenance and same-field evaluation limitations must remain in the manuscript methods and data documentation.

After training, run `report.py`, `summary_table.py`, `build_review.py`, `export_reference.py`, `reload_all.py`, `verify.py`, `figures.py`, `axis_figures.py`, `montage.py` and `package.py` in that order, with their displayed `--help` arguments. Verification requires the browser check report; `browser_checks.cjs` is the actual Chromium/Playwright test used in this environment (default localhost port 8876). Install `playwright-core` in a writable directory and expose it through NODE_PATH. Browser execution is optional for inference. This source file records the exact tested environment paths.
