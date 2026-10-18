# B3 temporal v12: root-segment coherence

T1 is baseline. T2 is relative to T1, T3 to T2, T4 to T3. Existing root portions are persistent; only an added tip portion is tip growth. A newly observed lateral root is new. Death is excluded. Connected components and crossings do not establish root identity.

## What is actually trained

Three separately continued 25-channel temporal CNN heads use the frozen U-Net, original TensorFlow DeepLabV3+, and SegFormer-B0 root segmentations, respectively. This is a comparison of three frozen root detectors plus temporal heads, not end-to-end temporal retraining of three backbones. Only the four upper B3 images (2541 × 1108 native pixels) are used. Previous and current RGB have model-predicted background removed. Human previous masks never enter prediction.

New supervision pools evidence along same-class runs of human root skeletons and penalizes within-run probability variation. A run ends at a manual class change or skeleton junction; these boundaries are never erased in the training labels. Existing pixel CE, class Dice, boundary and short-range consistency losses remain. All three heads undergo 40 continuation epochs with 48 paired draws/epoch, plus any accepted final refit. Raw trials and rejected checkpoints are retained.

At inference a diagonal-corrected skeleton graph is extracted from the predicted current root mask. Degree ≥ 3 nodes split chains. Mean log probabilities are pooled over root width and a three-state dynamic program penalizes label switches along each chain. It can retain real persistent-to-tip-growth changes. Branch/crossing pixels keep raw classification. Separate incident paths are not joined, and a whole connected network is never forced to one class. This suppresses short unsupported flicker; it does not prove root-instance identity, solve crossings, or guarantee every remaining transition is biologically correct. CNN probabilities retain inherited v11 class adjustments and are not altered by chain decoding to imply certainty in decoded labels. Confidence PNGs give the pre-decoding probability of the assigned class, not a structured posterior.

## Selection and assessment

As in the preceding experiment, spatial columns C0–C2 train, C3 selects checkpoints and decoding, and C4 is an already inspected repeated same-position benchmark. Sixty-four-pixel guards separate columns but image context and biology remain correlated. Four original RGB images give 12 train date-region views, four validation views, and four benchmark views; overlapping crops and augmentations do not create independent images. There are zero externally independent labelled test images. The same spatial region from all dates always stays in its group.

Candidate epochs 0,5,...,40 and chain penalties 0,.5,1,2,4,8,12 are evaluated on C3 only. A candidate is eligible if each strict class Dice is no worse than v11 minus .005, macro Dice no worse than v11 minus .002, and along-chain switch count does not increase. Rank eligible candidates by macro Dice + .03 × fractional reduction in switches. Epoch 0 with zero penalty remains a truthful fallback. Fewer switches are a smoothness measure, not proof of higher accuracy: true temporal class boundaries also count. All class confusion changes, including worse cases, and C4 results are retained. C4 does not choose the candidate. This repeatedly used C3 is development validation, not a blind estimate.

Evaluation starts from the old spatial evaluation head. The final deployment refit starts separately from the old all-upper final head and runs the selected number of epochs using all upper labels; it never supplies held-spatial scores. Full four-period maps are all-upper fitted visualizations, not independent predictions. Root masks and RGB remain unchanged. The frozen first-stage misses cap temporal recall, especially growth. Do not claim Dice > .9 unless measured. Original annotation SHA256 and every source label are verified unchanged.

Precision, Recall, Dice, IoU, confusion matrices, native maps, class/skeleton lengths and changes from v11 are included. Root lengths are human-mask skeleton comparisons in native pixels, not separately measured lengths or physical units. R² is identity agreement 1−SSE/SST, and negative values are retained. Correlated subregions/date pairs are not independent biological replicates. No significance or inter-tube generalization is asserted.

## Reproduce

Use the inherited environment (Python3.12, torch2.8, numpy2.2.6, scipy1.16.2, Pillow11.3, scikit-image.25.2, matplotlib3.10.6). CPU BF16 is used when available, with FP32 parameters and loss; CUDA is detected automatically. Keep the existing checkout. No web server is required.

From the repository root, with RUN=results/B3_temporal_v12, ROOTS=results/B3_three_model_v9, OLD=results/B3_temporal_v11, SRC=analysis-scripts/0353-2-B3_segmentation_pilot/temporal_v12:

```bash
.venv/bin/python "$SRC/run_all.py" --data "$RUN/data" --roots "$ROOTS" --previous "$OLD" --output "$RUN" --epochs 40 --samples 48
.venv/bin/python "$SRC/select_chains.py" --run "$RUN" --roots "$ROOTS" --previous "$OLD"
.venv/bin/python "$SRC/deliver.py" --run "$RUN" --roots "$ROOTS" --products
.venv/bin/python "$SRC/compare.py" --run "$RUN" --previous "$OLD"
.venv/bin/python "$SRC/audit_chains.py" --run "$RUN" --previous "$OLD"
.venv/bin/python "$SRC/verify.py" --run "$RUN" --roots "$ROOTS"
.venv/bin/python "$SRC/export_focus.py" --run "$RUN"
.venv/bin/python "$SRC/figures.py" --run "$RUN"
.venv/bin/python "$SRC/figures_compare.py" --run "$RUN" --previous "$OLD"
.venv/bin/python "$SRC/figures_chains.py" --run "$RUN" --previous "$OLD"
```

For another already registered position, run each frozen root detector first using the retained v9 scripts and weights. Then run predict.py with --head, --rgb-dir, --root-dir, --output. Four same-size RGB and corresponding root masks/probabilities are required. No human targets are needed. Original uploaded DeepLab source and inference are retained without silently substituting a PyTorch model. Unchanged large root probabilities and TensorFlow weights are linked by immutable commit and checksums in download_manifest.json.

Training-data ZIP: original full human JSON, upper RGB and unchanged targets, root masks, actual sampled native paired crops and augmentation logs. Weights/script ZIP: all three temporal trials/weights, v11 parent heads, full reproducible source and root model inference. Paper figures: 600 dpi PNG/TIFF plus PDF, quantitative SVG, captions. Model result ZIPs: full upper four-period maps, isolated original-root RGB, transparency, classes, original RGB, pre-decoding class probabilities and assigned-class confidence, and actual C3/C4 assessment.
