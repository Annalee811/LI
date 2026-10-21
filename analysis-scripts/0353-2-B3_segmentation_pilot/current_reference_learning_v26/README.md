# Training figures with the current annotation reference

Every displayed training/validation loss and validation metric uses the same
user-confirmed current annotation version. The200training views are identical
to the targets used to fit the three reported final post-annotation models.
The20validation views now use that version too. The24test views and reported
test probabilities remain unchanged. All dates at a coordinate retain one split.
These are correlated crops of four original images at one position, not244
independent acquisitions. External independently annotated test images:0.

`prepare.py --run ... --source results/B3_annotation_refinement_v1 --trajectory
results/B3_binary_continuation_v25_20261021` verifies and materializes the
current-reference inputs once; finished preparation is reused after hash checks.

The retained reported final models have test Dice0.884597820539224,
0.733292673082615 and0.7255330756354242. Their current-reference validation
Dice is0.8133791181161191,0.6673410363447264 and0.5937614463833092.
The split differs; validation and test Dice must not be substituted for each other.
The raw numeric scores are recomputed from unchanged saved probabilities and
current masks by `reference_scores.py`. No test score selects new weights.

The old training files did not archive current-reference validation measurements
at every epoch. Those missing records cannot be reconstructed by relabelling a
figure or rescaling final scores. This run performs an actual measured replay:

- Epochs1–20 follow the established current-target fine-tuning trajectory from
  its archived initialization, seed20261018,32batches×2draws per epoch.
- At index21 the reported final current-target checkpoint is restored, the
  optimizer restarts, seed20261021 and the queue reset, reproducing the recorded
  ten-epoch continuation. This stage boundary is marked on the figures.
- Rates and sample draws remain fixed to their recorded values inV25, preserving
  the established optimization schedule. Current-reference validation does not
  alter that schedule or replace the already reported final checkpoints.
- Actual train/validation losses, current-reference counts, Dice,IoU,Precision,
  Recall,accuracy and balanced accuracy are saved every epoch. No interpolation,
  manually increased score or synthetic loss is used. A partial monitoring epoch
  is64augmented draws, not a full200view sweep; total960optimizer updates/model.
- An internal old-reference consistency audit is retained in local histories
  only. It is never a training target, plotted curve or current accuracy result.
  Public plot-data tables omit those internal audit fields.

Every completed epoch now has a model-state snapshot. Torch stores FP32 native
state dictionaries; TF stores FP32 model variables and `weight_names.json`.
The latest resume checkpoint also stores optimizer,RNG,queue,history and draws.
Local resume pickles are trusted run artifacts; do not load untrusted pickles.
Completion guards reuse finished work. Existing source models and old results
are preserved. Current masks are frozen files, not regenerated from predictions.

`data` points to a byte-identical materialized view at `current_data_files`.
Materialization resolves legacy host-absolute RGB symlinks so the same inputs
work on the host and under the Docker `/work` bind mount. Every input hash and
split was checked against the current-reference manifest before training.

Host example; use `--model segformer` for the other Torch model:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B \
analysis-scripts/0353-2-B3_segmentation_pilot/current_reference_learning_v26/monitor_torch.py \
--run results/B3_current_reference_learning_v26_20261021 \
--source results/B3_annotation_refinement_v1 \
--trajectory results/B3_binary_continuation_v25_20261021 --model unet --threads 1
```

TF1.12 ResNet-v2-50 DeepLabV3+ uses the existing `b3-tf112-mkl:local` CPU image,
offline, through `container_entry.sh`, with repo mounted at `/work`,
`--tmpfs /ram-python:rw,exec,size=1300m`,OMP/MKL4and intra-op1. Supply
`--run`,`--source`,`--trajectory`,`--vendor results/B3_uploaded_deeplab_v1/vendor`.
No dependency installation, architecture substitution or GPU change is needed.

`finalize.py` waits for all three completed monitors, verifies90model snapshots,
draw order,rates,count-derived metrics and the unchanged reported checkpoints,
then calls `figures.py` once. The output includes a three-panel loss figure and
a six-panel loss/Dice/IoU composite,600dpiRGBLZW TIFF, with all30measured epochs.
The already verified current-reference test-score,confusion-matrix andROC/PR
TIFFs are reused byte-for-byte. No old-reference curve is displayed.

The published plot tables also support redraws without any training or model
loading: run `figures.py --run PATH_TO_DOWNLOADED_ARTIFACTS --table-only`.
Keep the delivered `report/figures` test TIFFs, current-reference table,
score table and `data/manifest.csv` together. The five learning/comparison
figures are regenerated from real saved values; the three unchanged fixed-test
TIFFs are hash-checked and reused. No missing value is manufactured.

These are new replay measurements, not a claim that previously unrecorded
historical validation losses were recovered. Replaying floating-point training
can produce numerical differences; loss/audit differences are retained rather
than selected to favor a model. The underlying annotation provenance remains
archived and traceable; user confirmation does not change its original source.
