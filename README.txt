V90 supplementary actual experiments — 2026-11-23

This archive adds internal temporal-input controls, retrained module ablations and a
paired structure-decoder study. It does not replace the original V90 weights.
15 runs = 5 variants x 3 fixed optimization seeds x 100 full-graph updates;
8192 sampled training root pixels per update. Checkpoint selection: validation only.
Upstream root segmentation and V66 priors are frozen, historically B3-exposed.
These are conditional internal B3 results, not independent cross-tube accuracy.

Main internal macro Dice: full 63.71 +/- 0.35%; frozen V66 62.56%.
No ConvLSTM 63.42%; no GAT 62.95%; no learned identity gates 63.69%.
The current-frame classifier collapsed to persistent-only under this fixed budget.
It is a negative control, not evidence about a fully tuned strong single-frame model.
Seed SD is optimization variability, not biological replication; no P-values invented.
Scores use eligible typed root pixels T2–T4, fixed foreground; background excluded.
Historical V90 references have a different training history and are not matched ablations.

Paired decoder study (separate whole-field inference context, same outputs per seed):
raw 63.70 +/- 0.34%; axis 73.51 +/- 0.50%; axis+junction 76.71 +/- 1.99%.
Instance pair precision ~95.91%, recall ~49.77%; overfragmentation remains.
Do not mix these whole-field stages into the partition-isolated main protocol.

Independent cohort: 3 tubes, 7 positions, 26 frames. Training tube 0353-2 excluded.
No external human temporal GT exists yet. Accuracy is NOT_EVALUATED_NO_HUMAN_GT.
390 new neural masks plus 52 frozen-reference masks are in the separate prediction ZIP.
Raw class codes: 0 background, 1 persistent, 2 newly observed, 3 extending,
4 first-observation root. Encoded masks look dark in a normal viewer; use the preview PDF.
PNG/native dimensions are preserved. Missing dates are not synthesized.
The experimental transfer adapter has documented geometric approximations.
External RGB had prior unlabelled prediction exposure; this is not a never-seen-image trial.

REPRODUCING CONDITIONAL TRAINING (Python 3.12, PyTorch 2.8 tested on CPU)
  python -m pip install -r requirements-experiments.txt
  python code/reproduce_study.py --output /a/new/empty/study
Uses bundled frozen features, current class GT subset and training-only normalization.
15 best and resume checkpoints, full histories and configs are under experiments/.
Original executed study_model.py and train_study.py retain their recorded SHA256.
The reproduction helper stages one historical baseline-image path to bundled data.
Large derived inference arrays are retained locally and can be regenerated from weights.

REPLAYING INDEPENDENT PREDICTIONS
Unzip Independent_inputs_*.zip beside this study; each supplies positions under
independent_inputs/. They can be extracted into the same directory.
  python code/reproduce_independent.py --inputs-root /path/independent_inputs     --model-root . --output /a/new/empty/predictions
This replays the raw heads from prepared features, not a standalone raw-RGB deployment.
Original raw-RGB upstream entry points and weights remain in the existing project.

REPLAYING THE PAIRED STRUCTURE DECODER
Unzip Decoder_inputs.zip, then:
  python code/reproduce_decoder.py --inputs /path/decoder_inputs     --model-root . --output /a/new/empty/decoder
All three stages share identical neural outputs within a seed. Human GT is not read
at prediction. Decoder inputs include existing B3 registered RGB, graph and topology.

FUTURE INDEPENDENT SCORING (not run as accuracy while human GT is unavailable)
Use the separate per-tube human-label kit; its all-255 masks are EMPTY TEMPLATES.
Fill from RGB independently, set annotation_state=human_complete and supply annotator
and annotation_source in a copy of reports/annotation_checklist.csv. Keep the locked
26 observations and hashes; no prediction-based cherry-picking or relabeling as GT.
  python code/evaluate_independent.py --manifest completed_checklist.csv     --annotation-root /path/annotation-kit --prediction-dir /path/to/one/model/seed     --output /a/new/evaluation
T1 is binary-root only. T2–T4 count human-valid background too. Results are grouped
by whole tube. Pending templates are rejected, not scored as all-background or zero.

FIGURES AND TABLES
reports/ contains a complete Chinese TXT report, XLSX and CSV metrics, two 600-dpi
Times New Roman figures/PDFs, and a seven-page external RGB-overlay preview.
Plot sources require a locally licensed Times New Roman installation; configure
the font directory F in plot_and_tables.py / plot_decoder.py when reproducing figures.
Font files are not redistributed. Test labels are not needed to regenerate predictions.

Original source: https://github.com/Annalee811/Temporal-segmentation-model/tree/4615bcbd37442a3b11e5ad4533242df95f237fb6
Canonical dataset policy: Q1-1-Z2 is the old 0353-1-Z2 alias; Q1-3-Z3 excluded.
No original files, weights, labels or earlier results were deleted or overwritten.
