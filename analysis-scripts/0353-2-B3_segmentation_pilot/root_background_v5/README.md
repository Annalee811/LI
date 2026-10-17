# 0353-2-B3 root/background stage

This stage supersedes the joint temporal trial: current RGB only, root/background
only, unchanged human foreground labels. Temporal attributes are preserved as data
but are not inputs or targets in this stage. Death is excluded.

Extract the data and trained-model ZIPs into the same workspace: `dataset/`,
`source/`, `unet/`, `segformer/`, `deeplab/`, `vendor/`. The separate initialization
ZIP uses `initialization/` to avoid overwriting trained weights. Run commands from
this workspace; `RUN` can be this workspace when using the exported trained models.
For a new training run, keep a separate output directory and copy the dataset there.

## Data protocol

The user declared the four upper regions completely annotated. Decode the original
progress JSON losslessly, use rows `[0,1108)`, interpret unpainted64 as background
only in this declared scope, retain128 as ignore, and retain255 as root. Do not
dilate masks or resample RGB. Original JSON retains every layer and lower labels.

Five448-pixel columns begin at22,534,1046,1558,2070. C0–C2 train; C3 validation;
C4 same-position spatial holdout. Cross-split guards are64 pixels. Ten overlapping
256-square windows per column ×4 periods yield120/40/40 patch images, corresponding
to30/10/10 physical windows. All periods at a window remain in the same split.
Four source images at one location are not200 independent observations. C3/C4 were
also used in earlier pilots; there are zero fresh external test images/locations.

During training, each epoch samples48 native crops, half fixed-grid and half
root-centred, within training columns only. Evaluation stitches the fixed windows
within each column, then counts each native pixel once. No held-out image context
is used outside the held-out column.

## Reproduction

Use Python3.12, torch2.8.0, torchvision0.23.0, transformers4.57.1, NumPy2.2.6,
Pillow11.3.0, scipy1.16.2, scikit-image0.25.2, matplotlib3.10.6. Torch trainers
automatically detect CUDA. The current machine has no CUDA and all runs use CPU.

`prepare.py --progress original_human_progress.json --originals ORIGINALS --output RUN`

`verify.py --run RUN --rgb-only-checkpoint OLD_UNET`

`train_torch.py --run RUN --model unet --initial OLD_UNET --epochs 90 --threads 1`

`train_torch.py --run RUN --model segformer --initial OLD_SEGFORMER --epochs 90 --threads 1`

Initial root-only models are included in the separate initialization archive. Their recorded source
hashes are in each final `training_run.json`. They were trained on the previous
T1/T2 labels, using the same training columns; C4 labels were not fitted.

DeepLab uses the uploaded original TensorFlow1.12 implementation, not a different
framework approximation. Run in the supplied `b3-tf112-mkl:local` Docker image:

```
docker --host=unix:///var/run/docker.sock run --rm --user 1000:1000 --cpus 2 \
  --mount type=bind,src=REPO,dst=/work -w /work \
  -e OMP_NUM_THREADS=2 -e KMP_BLOCKTIME=0 b3-tf112-mkl:local \
  python analysis-scripts/0353-2-B3_segmentation_pilot/root_background_v5/train_tf.py \
  --run RUN --vendor VENDOR --initial OLD_DEEPLAB/best_weights_fp16.npz \
  --epochs 36 --threads 2
```

`VENDOR` is the user's unmodified uploaded MIT source. Model: ResNet-v2-50,
output stride16, ASPP256 rates6/12/18, low-level48, decoder256,2softmax classes.
BN moving statistics are frozen from the root-only initial checkpoint, while all
convolution and affine weights are fine-tuned. All317 original variables are saved
FP16 in NPZ and restored into float32 before final validation/threshold/test.

The loss is0.5positive BCE +0.5hardest20%negative BCE +soft Dice +0.15soft clDice.
Training-only rotations/reflections and mild brightness/gamma augmentation do not
change stored RGB. Tiny U-Net and SegFormer train up to90epochs, original full
DeepLab up to36epochs; CPU compute budgets differ and are declared. All use the
same data and selection pixels, with the same initial BCE/Dice/clDice loss family.
The additional U-Net stage changes the hard-negative BCE weighting as detailed
below. No equal-compute or identical-final-objective claim is made.

An additional RGB-only U-Net refinement addresses thin-root boundary precision.
First preserve the completed first-stage model and its JSON/CSV metadata in
`RUN/unet_stage1_initial`. Then train with `--model unet --initial
RUN/unet_stage1_initial/best.pt --output-name unet_precision_candidate --loss
pixel-hard --lr 0.0001 --epochs 60 --threads 1`. This loss weights each positive
and mined negative pixel equally, using up to4×positive+256×batch hard negatives,
with the same Dice/clDice terms. `choose_unet.py --run RUN` compares calibrated
validation strict Dice and preserves the better binary model. This is one declared
validation-only additional U-Net trial, not equal optimization budget for all
three architectures. Both stages exclude temporal attributes and C4 fitting.
`unet_optimization.json` records both configurations, scores and hashes; the
separate initialization ZIP includes the exact first-stage checkpoint.

Select checkpoint by validation strict Dice at threshold0.5. Calibrate its root
threshold on validation0.1..0.9 step0.025. Freeze all checkpoints/thresholds and
validation-only model ranking before generating or scoring C4 predictions:

`evaluate.py --run RUN --mode select`

Then run each trainer again with `--evaluate test` and `--evaluate train` (same
required model/initial/vendor arguments). Only the selected model needs
`--evaluate upper`. Finally: `evaluate.py --run RUN --mode report`.

## Interpretation and gate

Primary metrics are foreground Precision, Recall, strict pixel Dice and IoU;
background-dominated pixel accuracy cannot establish root performance. Topological
clDice is a separate diagnostic, without dilation or tolerance. The user's strokes
are often1–3pixels wide; unchanged-mask agreement does not establish full anatomical
root-body boundaries. Unpainted fine roots and uncertain stroke placement remain
limitations. No new human truth is manufactured.

Require BOTH validation and spatial holdout root Dice≥0.95, Precision≥0.90,
Recall≥0.95 before starting temporal work. `stage_gate.json` records actual scores
and reasons. No time stage or other-position prediction is performed here, even
if the root gate passes; it needs a separate implementation and validation.

Reference lengths are skeletons derived from human masks, not independently
measured manual lengths. Pixel units only. Diagonal edge length is sqrt2, orthogonal
length1, with triangle shortcuts suppressed. Length field is computed before
partitioning into8blocks, each repeated4times. An isolated skeleton pixel counts
as1pixel by convention. Skeletonization can alter tiny corners and is not an
exact continuous curve-length measurement. R²=1−SSE/SST; negatives retained.
These32 descriptive pairs are correlated repeated observations, not32 biological
replicates. Report RMSE, MAE, bias, CCC, Pearson correlation and Bland–Altman too.

Whole-upper products include training pixels and are qualitative development
outputs. Official held-out metrics use native isolated columns only.

`label_audit.py` characterizes whole-upper human stroke geometry, including Dice
against a one-pixel-shifted copy of the same mask. This is a label-sensitivity
diagnostic, not prediction performance, an anatomical accuracy ceiling, or a reason
to relax the gate. It never changes labels or selects a fitted model.
