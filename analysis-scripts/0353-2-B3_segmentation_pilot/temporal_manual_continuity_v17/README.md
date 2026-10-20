# 0353-2-B3 — V14-based manual temporal continuity refinement (V17)

V14 is preserved. V16 is not used as the reference or warm start. The three V14 temporal heads are the starting networks; the binary root detectors remain frozen. This run addresses NEW/GROWTH class mixing along putative roots and wrong connections at crossings, while aligning reference width, boundaries and extent with the frozen U-Net binary root output.

## Reference and grouping

The exact complete original manual project is retained (SHA256 `27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09`). Only the registered 2541×1108 upper RGBs are used. RGB pixels/colors are unchanged. T1 is baseline; T2, T3 and T4 refer to their immediately previous date. No death class is included.

Reference root support equals the frozen U-Net root mask. Every retained originally typed root pixel keeps its original manual temporal class. Added width/tip pixels may inherit a nearby manual class within 8 native pixels; class-boundary ambiguity and model-only roots with no reliable manual class remain explicitly untyped (`255` in model rasters; root with temporal0 in the editor). No U-Net temporal prediction replaces the manual semantic layer. Original strokes outside model support remain in the original project and audit. Coverage and added/removed/unknown pixel counts are saved; this model-adjusted morphology is not a new independent human truth.

The same 25 base training windows, 25 shifted training views, 5 validation windows and 6 internal test windows are used for every model/date. All four dates at one window share a partition. There are 200/20/24 RGB views or 150/15/18 adjacent-date pairs. Cross-split pixel overlap is zero. This is one-position historical development data: earlier weights have same-field exposure, and whole-root prediction context can cross window borders. External independent test images: **0**.

## Learned routing and training

For each root detector separately, a small regularized logistic junction-affinity model learns from original manual class evidence inside **training windows only**. NEW versus persistent/growth arms provide negative continuity evidence; the same class and straight geometry provide weak positive evidence. Input features are direction at 12/24/40 nodes, root RGB/width, and predicted previous-axis support. Same-class geometry is weak supervision: original raster labels contain no biological root-instance IDs. Training fit accuracy is not independent root-identity accuracy.

The learned affinity refines opposite-arm pairing at crossings; side roots remain separate and ambiguous crossings are flagged. Full-frame root axes and previous-date correspondence are computed before cropping. Each 31-channel paired temporal head starts from its own V14 25-channel checkpoint, with six additional context-channel weights zero-initialized. All three heads train for **32 epochs / 512 updates** with identical paired draws and augmentations. The class-calibration multipliers inherited from V14 are preserved in the saved checkpoints.

AdamW: learning rate1e-4, weight decay1e-4, cosine decay to1e-5; batch4,64 draws/epoch. Class weights are inverse-square-root frequency ratios capped at8. The objective combines weighted CE, class Dice, semantic-boundary BCE, local same-class consistency, same-class skeleton-run loss and an additional0.25 routed same-manual-class consistency term across junctions. True manual class transitions are not smoothed together. The root networks stay frozen: U-Net/SegFormer binary networks use PyTorch; the supplied DeepLabV3+ binary model uses TensorFlow1.12. The paired temporal heads use PyTorch. This is a two-stage model family, not newly end-to-end-trained temporal versions of three backbone architectures. CUDA is detected automatically; this execution uses CPU BF16/FP32.

Checkpoints are selected on full-frame strict-axis validation macro Dice every4epochs, with epoch0 eligible. Post-fit selection uses the **same15validation pairs only**, evaluating72 fixed combinations: new/V14 emission mixture weights [0,.5,1], growth factors [1,2,4], new-root factors [1,.75], and four decoder profiles. The final learned-emission profile allows an axis to be NEW, or terminal GROWTH / PERSISTENT / terminal GROWTH. Unlike the strict profile, a binary previous-tip miss does not erase all learned extension evidence; uncertain all-growth fragments require review. Original audit/test scores do not select a profile.

Final selections are recorded in each decoder JSON. U-Net uses the new temporal head (epoch16), growth factor2,new factor.75,soft prior.5. DeepLabV3+ uses V14 emissions (new-head mixture0), factor1/1,soft prior0. SegFormer uses V14 emissions (mixture0), growth factor2,new factor1,soft prior0. All three new heads were genuinely trained32epochs (selected epoch16 each), but validation did not favor the DeepLab/SegFormer new emissions. Their improved delivered pipelines use the learned routing/decoder, while retaining V14 classification emissions. Do not attribute all changes to successful backbone fine-tuning.

## Evaluation

`V14_published_metrics.csv` preserves the user's V14 table: revised-reference macro Dice .797955/.474196/.523801; original-audit .474648/.361412/.358614. Those scores use the V14 teacher reference and tilewise inference. They are **not directly comparable** with scores on the new manual-class/morphology reference.

`V14_matched_comparison.csv` compares both versions using the same saved full-frame crops and each of three references (manual morphology, exact original human, V14 teacher). End-to-end Precision/Recall/Dice/IoU pool confusion counts over18test date-pairs before averaging the three temporal classes; background/T1 are excluded. Unknown reference pixels are ignored explicitly; their coverage is disclosed. Undefined-precision classes with known support have zero precision in the summary. Full fields include fitting pixels and are qualitative.

`root_continuity_before_after.csv` counts NEW/GROWTH coexistence and short class runs on common putative axes, excluding crossing nodes owned by a different axis. Zero NEW/GROWTH coexistence is **an enforced constraint, not independently measured biological identity accuracy**. Junction-neighborhood Dice measures temporal class agreement, not which physical root crosses above another. Review masks, root-axis IDs and axis-match records are retained. Independent instance annotations are not available, so ambiguity at crossings is not claimed solved completely.

Skeleton pixel-count agreement, OLS R², identity R², RMSE, MAE and CCC use18correlated tile/date pairs. These are not independent manual physical-length measurements. No pixel-to-mm calibration is invented.

## Reproduction

Extract desired ZIPs into one parent directory; repository paths are preserved. Install `requirements.txt` in a fresh Python3.12 environment. Prepared data can be reused; do not overwrite the delivered run. Required inherited v13/root/TF sources, exact V14 starting heads, routers and frozen root caches are supplied.

```bash
SRC=analysis-scripts/0353-2-B3_segmentation_pilot/temporal_manual_continuity_v17
RUN=results/B3_temporal_manual_continuity_reproduce
ROOTS=results/B3_annotation_refinement_v1
BASE=results/B3_temporal_reference_v14
python "$SRC/prepare.py" --run "$RUN" --roots "$ROOTS" --previous "$BASE"
python "$SRC/setup_routes.py" --run "$RUN" --roots "$ROOTS" --previous "$BASE"
python "$SRC/run_all.py" --run "$RUN" --roots "$ROOTS" --previous "$BASE"
for MODEL in unet deeplab segformer; do
 python "$SRC/calibrate_continuity.py" --run "$RUN" --roots "$ROOTS" --previous "$BASE" --model "$MODEL"
done
python "$SRC/report.py" --run "$RUN"
python "$SRC/compare_v14.py" --run "$RUN" --previous "$BASE"
python "$SRC/summary_table.py" --run "$RUN"
```

If the old V14 feature cache is absent, `cache.py` recomputes the prediction-only 25base channels from RGB/root probabilities. Training labels never enter inference features. `predict_RGB.py` requires selected head, V14 baseline head, learned router, decoder JSON, four aligned RGBs and frozen root outputs. `reload_all.py --run ... --roots ... --previous ...` rebuilds routes/features from RGB without reading labels and checks all four saved predictions for each model pixel-for-pixel.

`build_review.py` makes the standalone four-date `annotate.html`; `export_reference.py` saves its native reference overlays and original-color transparent roots. The editor supports Ctrl+Z, fixed tools,60×zoom and zoom retained across dates. It was tested in Chromium with real brush/undo/save/reimport actions. The exact original project is separate.

Use `figures.py`, `axis_figures.py`, `junction_figures.py --previous ...`, `montage.py --output ...` and `finish_tiffs.py` for600dpi figures; `verify.py` audits final labels, weights, splits, metrics and actual label-free reloads. `package.py` builds and CRC-checks all archives. A development-only preliminary run and pre-calibration outputs are retained locally but excluded from delivered training/results.

## Figures and interpretation

The integrated TIFF is a single A3 landscape page,600dpi,72panels: four date blocks, original full/local RGB controls, manual/morphology reference, three final model outputs, full/local overlays and background-removed temporal roots. Blue baseline,purple persistent,green new,gold growth; gray untyped reference. Native root RGB cutouts are also in results. Display scaling does not create new native detail.

Supplemental S7 shows V14 versus current crossing details in identical held-out windows selected by V14 NEW/GROWTH content, without using current scores to choose. S8 counts the resulting consistency constraint. S1–S6 show actual training histories, transition Dice, pixel-length agreement, window variability, local class errors and confusion matrices. Root-axis inspection is qualitative and does not establish biological IDs. Keep reference provenance, same-field limitations and the difference between routing constraints and independent accuracy in manuscript reporting.
