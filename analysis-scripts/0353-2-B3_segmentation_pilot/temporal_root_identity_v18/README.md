# 0353-2-B3 — joined-root temporal review and selective correction (V18)

This version addresses the user's two T2 details where a visible root still switched among persistent, newly observed and extension labels. V14 and V17 are retained. The exact original human JSON is untouched. Root morphology and native RGB are unchanged from V17. This is a model-supported, reviewable refinement, not independently established biological root identity.

## What changed

V17 constrained labels inside its extracted axes, but one physical root could still be split into several axes or incorrectly paired with a nearby root at a dense intersection. This version uses multi-scale tangent, width, RGB appearance and entry/exit lateral position. Junction pairings maximize supported pair benefit instead of forcing the greatest number of pairs. Learned semantic affinity is a small prior and cannot veto a strongly supported geometric continuation. Well-separated fragment endpoints are stitched using geometry, width, appearance and existing foreground support. No RGB/root-mask pixel is filled.

Temporal decoding operates on the resulting full-image candidate roots, before publishing crops. A candidate root is entirely newly observed, or a persistent shaft with terminal extension. Growth is excluded from an observed old shaft and from a newly observed root without an old-tip continuation. Previous predicted-root visibility provides the birth/continuation evidence. Unresolved intersections remain flagged. Single-root panels and exact per-axis class runs expose overlapping root paths; a 2D raster alone cannot assign two different classes to the same crossing pixel.

## Conservative annotation revision

Only U-Net/preceding-human-supported edits are applied. A persistent old-core correction requires current U-Net root score >=.8, previous U-Net root correspondence >=.8 and joint previous manual/U-Net support >=.7 on a sufficiently long root. A newly-observed correction requires current root score >=.8, previous model correspondence <=.1, absence from preceding manual traces >=.9, model new-or-growth score >=.8 and no old-tip continuation. Ambiguous and untyped pixels stay untyped. Original files and frozen V17 reference are separately preserved.

The final proposal changes **0 / 2142 / 885 / 554** temporal pixels in T1/T2/T3/T4, respectively, across120logged root events. Total:3581. Geometry/unknown masks do not change. These are differences from V17's morphology-adjusted reference; they do not count earlier width adjustments relative to raw manual brush strokes. Every change has a root ID, bounding box, class transition, model score, prior visibility and reason in `revision_log.json`. The teacher geometry and evidence sheets are included. `human_reviewed:false` remains until actual review. This corrected reference is not independent manual ground truth.

The teacher is the delivered V17 U-Net class score plus the new geometry, with preceding original human root traces used only for selective reference correction. Inference on new images reads no manual labels. The original complete JSON SHA256 is `27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09`.

## Training and selection

All three paired temporal heads were optimized together in two stages:32epochs in the first geometry/reference pass, then16epochs using final lane-aware geometry and final revised reference. Stage2starts from each model's selected stage1checkpoint, not necessarily its last epoch. Thus32+16describes executed training, not a claim that the selected weight followed48uninterrupted epochs. Stage1data, selected/last weights, histories and complete source snapshot are retained separately. All root/background networks stay frozen.

The heads have31input channels:25existing adjacent-RGB/predicted-root channels,4global root correspondence channels and2RGB contrast channels. Weighted cross entropy, class Dice, boundary BCE, local same-class consistency, skeleton-run consistency and routed same-class continuity are used. AdamW1e-4, cosine decay to1e-5,batch4,64paired draws/epoch; identical draws/augmentations across models. CUDA is detected automatically; this run uses CPU BF16/FP32. The DeepLabV3+ binary root network is the supplied TensorFlow1.12model, while the three paired temporal heads use PyTorch. This is not end-to-end temporal retraining of all three backbones.

Checkpoints and final emission mixing/class ratios/geometry strength are selected **only on unchanged V17 validation labels**. Final calibration uses36fixed combinations of warm/new emission weights [0,.5,1], growth factors [1,2], new factors [1,.75] and identity-prior strengths [.5,1,1.5]. Starting checkpoints are eligible. The selected head/blend is recorded; actually running extra epochs does not imply a better selected checkpoint.

In the actual final selection, all three new-head emission weights are zero. Final inference retains each selected stage1 emission head, together with the final router and identity decoder. The 16 additional optimization epochs did not improve the selected emission head. Against the unchanged V17 internal test reference, Dice is 0.6585 / 0.6021 / 0.5733 for U-Net / DeepLabV3+ / SegFormer-B0; corresponding V17 scores on identical crops are 0.6872 / 0.5042 / 0.5660. This is a mixed outcome, not a universal accuracy improvement. The corrected-reference scores are separately reported model-supported agreement values.

There are200/20/24RGB windows,150/15/18adjacent-date pairs, with all4dates at a spatial window grouped. Cross-split pixel overlap is zero. Root trajectories are traced over the full field, so context can cross crop boundaries. Earlier weights have same-field exposure, and user-reported error details are used for development. This is not a fresh blind test. External independent images:0.

## Evaluation and limitations

Three references are reported separately: exact original human labels, unchanged V17 morphology-adjusted labels (the PRIMARY comparison), and the selectively corrected proposal (model-supported agreement, not independent accuracy). No test label is replaced for the primary before/after comparison. Precision/Recall/Dice/IoU pool confusion counts over18correlated internal date-pairs and then macro-average persistent,new,extension classes. T1/background do not inflate the temporal macro. Known untyped pixels are ignored and their counts remain unchanged.

Continuity tables count NEW/PERSISTENT and NEW/GROWTH mixtures on common candidate roots. Zero is an enforced structural constraint, not100%biological identity accuracy. Review masks identify unresolved roots. Candidate track IDs are provisional cross-date associations and never classify an unmatched root as dead. Physical parent/lateral identity at some crossings remains unverified. No independent instance labels or physical manual root-length measurements exist. Skeleton pixel-count regression is not calibrated physical length.

A lower Dice against an unchanged reference can coexist with corrected routing or a model-supported label proposal; report both rather than replacing the denominator or claiming universal accuracy improvement. The updated reference needs user/expert review before being treated as ground truth. Root-stage morphology follows U-Net, so agreement against it favors that detector.

## Reproduction

Extract desired ZIPs into a shared parent; repository paths are retained. Prepared labels, root caches, routers, weights, exact teacher inputs and scripts are supplied. Install `requirements.txt` in Python3.12. Work in a new run directory and do not overwrite delivered results.

```bash
SRC=analysis-scripts/0353-2-B3_segmentation_pilot/temporal_root_identity_v18
RUN=results/B3_identity_reproduce
ROOTS=results/B3_annotation_refinement_v1
PREV=results/B3_temporal_manual_continuity_v17
STAGE1=results/B3_temporal_root_identity_v18_stage1
python "$SRC/prepare_teacher.py" --run "$RUN" --roots "$ROOTS" --previous "$PREV" --stage1-run "$STAGE1"
python "$SRC/prepare.py" --run "$RUN" --roots "$ROOTS" --previous "$PREV"
python "$SRC/setup_routes.py" --run "$RUN" --roots "$ROOTS" --previous "$PREV"
python "$SRC/run_all.py" --run "$RUN" --roots "$ROOTS" --previous "$RUN/warm_starts"
for MODEL in unet deeplab segformer; do
 python "$SRC/calibrate_continuity.py" --run "$RUN" --roots "$ROOTS" --previous "$RUN/warm_starts" --model "$MODEL"
done
python "$SRC/reload_all.py" --run "$RUN" --roots "$ROOTS" --previous "$RUN/warm_starts"
```

If the prior feature cache is absent, `cache.py` recomputes its25prediction-only base channels. No manual mask enters inference features. Stage1can be replayed using `temporal_root_identity_v18_stage1` source and its exact data/teacher evidence; alternatively use its provided selected starting weights to reproduce final refinement. Numeric results may differ across CPU/GPU arithmetic; actual CPU reloads are verified pixel-for-pixel.

`report.py`, `compare_v17.py`, `summary_table.py`, `candidate_tracks.py` produce tables. `build_review.py`, `export_reference.py`, `revision_review.py` make the editable project and evidence sheets. `figures.py`, `axis_figures.py`, `root_details.py`, `montage.py` and `finish_tiffs.py` generate600dpiTIFFs. `verify.py` audits original hashes, changes, frozen evaluation references, actual training, weight reloads and recomputed metrics; `package.py` CRC-checks archives. The original36KBeditor template, inherited v13sources and frozen binary model/TF runtime sources are included.

## Figures and editing

The integrated TIFF is one600dpiA3page with originalRGB, corrected reference and3model results for4dates, full/local overlays and background-removed roots. The two user details are also a separate TIFF. Individual candidate-root panels show one root at a time, including crossing paths. Arrows show orientation, not independently verified growth polarity. Bluebaseline,purplepersistent,greennew,goldextension,grayuntyped reference. Native images are not color-enhanced, resized or warped in the data files; figure scaling does not add native detail.

`annotation_review/annotate.html` embeds all4upperRGBs and the final reference. Chrome/Edge editing supports fixed tools,60×zoom,Ctrl+Z,date-switch zoom retention,save and reimport. Original human and V17projects and per-root revision evidence are separate. Browser brush/undo/save/reimport behavior is actually tested. Full-field panels include fitting pixels; quantitative comparisons use the documented internal windows only.
