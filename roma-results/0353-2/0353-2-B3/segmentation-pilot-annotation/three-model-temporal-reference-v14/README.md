# 0353-2-B3 temporal reference v14

Four native registered upper RGBs (2541×1108), T1 baseline and three adjacent transitions. This is a **versioned U-Net teacher-reference update and temporal-head continuation**, not a new independent human ground truth or an end-to-end temporal retraining of all three root backbones. Death is excluded. Original full annotation JSON is byte-preserved with SHA256 `27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09`.

## Biological labels and review

T1 roots are baseline only. At each later date, the reference is relative to the immediately preceding date: existing visible segment persistent, previously unobserved root/side root newly observed, and new tip segment growth/extension. A root may contain an old persistent part and an added growth part. Crossing or touching alone does not establish a single root's biological identity. Occlusion, root motion and residual registration errors can resemble appearance/disappearance; disappearance is not proved death.

The v13 U-Net all-upper temporal head, combined with the latest frozen binary U-Net support, produced the replacement upper-region reference. No per-root expert adjudication or biological identity guarantee is claimed. Changes, low-confidence pixels and both original reference versions are retained. `active_progress.json` and `annotate.html` allow further user review. UI temporal codes are **1 new / 2 growth / 3 persistent / 4 baseline**, whereas exported model classes are **1 persistent / 2 new / 3 growth / 4 baseline**. `prepare.py` performs this explicit conversion.

## Data and actual fitting

36 native nonoverlapping base 256-pixel windows: fixed random25training/5validation/6internal-test locations, seed20261018. Training adds25 shifted12-pixel views, excluding all validation/test pixels. The four dates at a location stay together. Thus200training RGB views,20validation and24internal-test RGB views yield **150/15/18 adjacent-date pair examples**. Shifted views overlap their own training bases; these are not independent images or additional plants. No difficult tile is filtered out. Confidence≥0.5 teacher root pixels on each student's own detected support supervise fitting; all typed reference pixels enter the end-to-end report.

Each of three frozen root/background networks (U-Net, original TensorFlow DeepLabV3+, SegFormer-B0) supplies predicted roots, probabilities and RGBs masked to those roots. A separate98668-parameter25-channel paired temporal CNN is continued for each backbone,24epochs,64draws/epoch,batch4,384updates. All three share identical paired draws and geometric/brightness augmentations. AdamW starts at1e-4,weight decay1e-4,cosine decay to1e-5. The inherited objective combines weighted CE,0.5classDice,0.2auxiliary boundaryBCE,0.05within-class consistency and0.3same-class skeleton-run loss. Previous masks are **model predictions**, never human labels at inference. Root support remains exactly frozen.

Checkpoints (including epoch0) are selected by revised-validation macroDice measured every4epochs. A fixed chain penalty grid0,.5,1,2,4,8,12 is then selected on the same15validation pairs; test scores do not select parameters. Skeleton decoding splits at junctions and penalizes rapid label changes; it never assigns one compulsory class to an entire connected root network. The selected epochs are24/20/12 and penalties1/8/2 for U-Net/DeepLab/SegFormer. Actual histories precede decoder calibration and are not reconstructed or cosmetically smoothed.

The delivered run used CPU BF16 autocast for temporal fitting/evaluation, FP32 objectives/weights, and CPU FP32 whole-frame inference. CUDA is automatically detected. Device arithmetic can cause minor reproduction differences. Full-frame prediction blends overlapping256px windows, whereas quantitative evaluation uses the saved256px test-window predictions; do not recompute published tile metrics by cropping blended full-frame maps.

## Evaluation meaning

Report Precision,Recall,Dice,IoU per class, with macroDice over the three nonbaseline temporal classes; T1 is not used to inflate temporal accuracy. Both original and teacher-derived references are supplied. End-to-end scores include frozen root-detection errors; conditional detected-reference-root scores are a separate diagnostic. Aggregate confusion counts first rather than averaging unequal per-image ratios.

**Revised-reference Dice is teacher agreement.** The teacher previously fit the same full upper human labels; warm-start evaluation heads used different historical same-field splits and may have seen current held-out pixels. Consequently even this round's nonoverlapping spatial holdout is not a new blind test, and neither reference comparison proves cross-position/cross-tube accuracy. Independent external test images:0. Original reference audits are also historical development data, not a pristine external benchmark. U-Net distillation favors the U-Net reference generator.

Skeleton pixel counts are compared for18correlated tile/date pairs. Identity R² measures agreement with y=x; OLS R² measures linear association and is reported separately. RMSE,MAE,bias andCCC are provided. These are mask-derived pixel lengths, not independent physical manual root-length measurements. No scale or physical length is invented. Whole-upper figures contain training pixels and are qualitative.

## Download packages

Extract all desired ZIPs into one parent directory. They share `0353-2-B3_temporal_v14/` and retain repository paths. Annotation kit: standalone `results/B3_temporal_reference_v14/annotation_review/annotate.html`, four images and labels embedded, fixed tools,60×zoom,Ctrl+Z,date-switch zoom retention,save/reimport tested. Each model result ZIP: four-period RGB overlays, transparent original-color roots, temporal color roots,classes,confidence/probability,exact validation/test predictions and frozen binary root caches. Data: exact paired tile targets,original/revised maps,RGBs and source JSONs. Weights: frozen root networks,new temporal heads,warm-start heads and teacher. Scripts: the complete v14 workflow plus required inherited temporal,root-model,identity and TF runtime sources. Figures:600dpi LZW TIFF,PNG,and vector PDF/SVG for numerical charts; large raster PDF duplicates are omitted from the downloadable figure ZIP.

## Reproduce the continuation without overwriting this run

Install `requirements.txt` into a dedicated Python3.12 environment. Extract data,all3model-result ZIPs,and the weight/script packages. From `0353-2-B3_temporal_v14`:

```sh
mkdir -p results/B3_temporal_reference_v14_reproduce
cp -a results/B3_temporal_reference_v14/data results/B3_temporal_reference_v14_reproduce/data
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_reference_v14/run_all.py --run results/B3_temporal_reference_v14_reproduce --roots results/B3_annotation_refinement_v1 --previous results/B3_temporal_v13
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_reference_v14/calibrate_decoder.py --run results/B3_temporal_reference_v14_reproduce --roots results/B3_annotation_refinement_v1
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_reference_v14/report.py --run results/B3_temporal_reference_v14_reproduce
```

Do not point fitting/calibration at the delivered published directory. `calibrate_decoder.py` refuses to overwrite its calibration backup. `prepare.py` documents the original reference generation; exact prepared references are already included. Rebuilding the web editor also uses the included36KB custom editor template. The optional TF1.12 runtime builder is supplied for frozen DeepLab root inference; **this temporal continuation uses Torch for all three heads and does not require rerunning TF**. Legacy runtime setup concerns in its older README do not apply to the already-cached root stage.

## Predict / regenerate four-period products

With the provided cached root outputs, for each `MODEL` of `unet`,`deeplab`,`segformer`:

```sh
python analysis-scripts/0353-2-B3_segmentation_pilot/temporal_v13/predict.py --head results/B3_temporal_reference_v14/MODEL/best_temporal_head.pt --rgb-dir results/B3_temporal_reference_v14/source_RGB/0353-2-B3 --root-dir results/B3_annotation_refinement_v1/MODEL/upper --output results/reproduced_predictions/MODEL
```

For another position first run the corresponding native-RGB root predictor in `annotation_refinement_v1`, then the temporal predictor. Use matched common-coordinate adjacent RGBs; do not fabricate missing periods or train on the excluded Q1-3-Z3. Canonical Q1-1-Z2 replaces legacy0353-1-Z2. No other position is predicted in this run. To regenerate tables/figures use `report.py` / `figures.py --run ...`. `verify.py --run ... --roots ...` independently reloads all3selected heads and reproduces all18test-pair confusion matrices, checks original hashes,root support,identical draws and zero cross-split overlap. `verification.json` records actual checks; `editor_verification.json` records real Chromium brush/undo/zoom/save/reimport checks.
