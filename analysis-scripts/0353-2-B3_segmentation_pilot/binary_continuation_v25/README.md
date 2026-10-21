# Binary-model continuation and loss monitoring

This run preserves the exact training data, fixed spatial/date allocation and
the existing three selected final models from `B3_annotation_refinement_v1`.
No annotation is regenerated. All four dates at a tile keep one partition.
There are 200 correlated training views, 20 validation and 24 internal-test
tile-date images. External independently annotated images: 0.

The old run did not compute or archive validation loss. The first 20 epochs
are therefore **replayed measurements**, not original recorded validation
losses. Starting-weight SHA256s, architecture code and training draw sequences
are checked against the old run. All epoch-wise differences from old losses
and available old Dice records are retained. A relative loss difference <=1%
and logged validation Dice difference <=0.005 are reported as numerical
proximity, never as proof of identical weights or original historical data.
Every model and epoch is reported even if the replay is worse. Values are not
interpolated, made from 1-Dice, manually perturbed, or selectively substituted.
Original histories and selected weights stay unchanged in the old run folder.

After this replay, each model reloads the exact **old final selected weights**.
Additional monitoring epochs start at index 21, with a fresh optimizer at
1e-5; the old optimizer states were not archived. The plot marks this stage
boundary. This is a warm restart from the selected checkpoint (old epochs
18, 12, 16), not a claim of bit-exact continuation of the old epoch-20 state.
Maximum cumulative index: 60; minimum additional epochs: 10; early-stopping
patience: 8 epochs. One monitoring epoch retains the old definition of 64
augmented draws / 32 batches of two, not a full pass through the 200 views.
Actual sample counts and optimizer updates are reported alongside epochs.

The original model remains eligible for retention. Replacement requires
original-validation Dice at the old frozen probability threshold to improve
by >1e-4, Precision/Recall to drop by no more than 0.005 relative to the old
baseline, and validation loss <= original baseline loss+0.01. Only validation
data affect model selection and LR adaptation; test results never decide
which weights are retained. The LR halves after four loss checks without an
improvement of 1e-4 (minimum 1e-7). Changes in test scores cannot be guaranteed
away. If the original model wins, its probabilities, masks and threshold are
reused byte-for-byte, avoiding unnecessary inference and preserving old scores.

Training and validation loss use the same data objective: hard-negative BCE
+ soft Dice +0.15 soft-clDice (five skeleton iterations). Optimizer weight
decay/L2 is not added to the plotted data loss. Validation uses unaugmented
fixed batches, FP32 eval mode and no BN updates. Actual losses, Dice, IoU,
accuracy, Precision/Recall, LR and counts are saved on **every** measured
epoch. Checkpoints save optimizer state, Python/NumPy RNG, Torch RNG where
applicable, queue and draw sequence after each epoch. TensorFlow 1.12 has no
stochastic network layers in this source architecture. Local resume.pt is a
trusted run artifact; verify hashes before loading a downloaded pickle.

Training-selection curves use the same original frozen validation labels as
the old training run. Final reporting-reference scores use the same revised
reference as the named final paper package and are reported separately from
the original-label audit. Teacher-influenced agreement is not an external
independent biological accuracy estimate. None of the test tiles are omitted.

Host Torch jobs use the existing environment; example:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -B \
analysis-scripts/0353-2-B3_segmentation_pilot/binary_continuation_v25/train_torch.py \
--run results/B3_binary_continuation_v25_20261021 \
--source results/B3_annotation_refinement_v1 --model unet --threads 1
```

Use `--model segformer` for the second Torch model. The genuine TF1.12
ResNet-v2-50 DeepLabV3+ runs inside the already prepared CPU image through
`container_entry.sh`, binding the repository at `/work`. Data/revision symlinks
are relative so they work on host and in the container. Supply `--run`,
`--source`, `--vendor results/B3_uploaded_deeplab_v1/vendor`, `--threads 2`.
The `--predict-only` path evaluates a newly selected TF checkpoint without
retraining. Completed training runs are not repeated.

The actual TF replay used intra2/OMP2 for epoch1, intra4/OMP4 for epochs2–3,
intra4/OMP1 for epoch4, and intra1/OMP4/MKL4 from epoch5. Each restart loaded
the last completed model/optimizer/RNG/queue checkpoint. See preparation.json
and the individual logs rather than assuming a single CPU thread setting.

After all three models finish, generate and audit the ten core figures
(plus one separate replay-audit TIFF for each unmatched replay):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
analysis-scripts/0353-2-B3_segmentation_pilot/binary_continuation_v25/figures.py \
--run results/B3_binary_continuation_v25_20261021 \
--source results/B3_annotation_refinement_v1
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
analysis-scripts/0353-2-B3_segmentation_pilot/binary_continuation_v25/audit_and_summarize.py \
--run results/B3_binary_continuation_v25_20261021 \
--source results/B3_annotation_refinement_v1
```

The running finalizer already generates the figures once; do not repeat that
command if report/completion.json and its hash verification passed. The audit
performs no neural-network inference or optimization. It verifies the replay
draw order, all epoch records, final checkpoints, byte-identical reused
predictions, all600dpiTIFFs and the full ROC AUC by an independent tied-rank
calculation. `all_epoch_loss_and_metrics.csv` contains every displayed loss;
`original_vs_replayed_training_losses.csv` preserves every comparison,
including unfavorable deviations. `replay_and_continuation_summary.csv`
records proximity flags, actual training draws and retained models.

If either advance replay-proximity check fails, its main loss plot retains
the archived original training loss for epochs1–20 and leaves the unrecorded
original validation loss missing. A separate supplementary figure displays
the full measured replay. All measured replay data remain in the tables;
no unfavorable epochs are silently omitted. primary_loss_plot_data.csv
identifies the exact provenance of each point in the primary loss figures.
