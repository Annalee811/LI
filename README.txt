THREE KEY FIGURES — TWO ROOT CLASSES (2026-11-23)

Existing = original persistent1 + extending3; New = original newly observed2.
This is a post-hoc two-class re-evaluation of the existing15 trained runs. Model
weights, original three-class training objectives and validation checkpoint choices
were not changed. No new two-output network is claimed to have been trained.
Class masks and truth were merged before metrics. Do not average old class scores.
New-class metrics remain identical; two-class macros cannot be compared directly
with three-class macros as if the change were a model improvement.

Figure1: current-frame negative control, frozen V66 pairwise reference,
retrained ConvLSTM without GAT, and retrained Full temporal graph. Internal B3
eligible root support only, background excluded. Frame-only collapsed to all
Existing, so New Precision is undefined. This is not a tuned strong single-frame
benchmark. The frozen prior had historical B3 exposure; this is conditional
internal evaluation, not independent generalization.

Figure2: matched module usage/effects plus a SEPARATE paired whole-field decoder
experiment. Three fixed seeds measure optimization variation, not biological n.
No p-values or significance claims are made. Memory removal retains other
historical features; gate removal retains GAT attention/geometric priors.
Grouping preserves root IDs; good class Dice does not prove correct root identity.

Figure3: locked3 tubes,7 positions,26 input images; training tube0353-2 excluded.
The19 post-baseline images still lack human temporal truth. All independent
metrics are NULL, not zero. Main examples use the raw retrained Full neural head
at the first fixed seed20261123; the complete axis/junction decoder is not
included in these external head predictions. The separate frozen V90 transfer
adapter (historical weights and documented input approximations) is shown in
the7-page full-position atlas, alongside V66 and the retrained head.
Prior unlabelled image/prediction exposure is disclosed; not a prospective
never-seen-image trial. Example positions are first canonical IDs per tube;
crop selection uses common binary foreground coverage, never GT or quality rank.

Files: three600dpi JPG/PNG/TIFF figures, font-embedded PDF, editable SVG,
Word captions/results, Excel full values, and442 native two-class predictions
(390 retrained head maps +52 frozen-reference maps). Source models/labels untouched.
Colors: gold New, blue Existing; gray baseline is unassigned, not a third root type.
T1 not scored; missing dates not synthesized. Original class2 pixels unchanged.

Numeric reproduction (NumPy,Pillow):
python reproduce_metrics.py --bundle . --output /new/empty/check
Saved held-out truth and binary predictions reproduce all26 confusion matrices.
The original15 trained best/resume checkpoints, logs and source are still at:
https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/V90_actual_experiments_code_weights_results.zip
Checkpoint hashes are retained in Existing_checkpoints.json.

Future independent evaluation:
Use Independent_two_class_annotation_checklist.csv. From RGB independently,
annotate0 background,1 Existing,2 New,255 uncertain. A New-vs-non-New binary
truth cannot distinguish Existing from background and must not be reused as such.
Set annotation_state=human_complete and fill annotator,annotation_source.
Keep annotation_schema=background_0_existing_1_new_2. Then:
python evaluate_independent_two_class.py --manifest completed.csv  --annotation-root /GT --prediction-dir /one/model/seed --output /new/evaluation
The full26-image cohort is locked; T1 is excluded from temporal scores. Ground
truth background IS included in independent class errors, unlike the internal
fixed-root evaluation. Score raw grouped predictions, not masks with rejected
pixels silently removed. Empty GT and missing human provenance are rejected.
Original3-class human labels can also be grouped with --gt-encoding original-three
and explicit schema background_0_persistent_1_new_2_extension_3.

All figure text is black Times New Roman, white background,8pt axes/12pt letters.
To redraw: python draw_three_key_figures.py --fonts /licensed/font/directory
Fonts not redistributed. Standard dependencies: numpy,Pillow,matplotlib,
PyMuPDF,openpyxl,python-docx. Dataset/current identity policy is unchanged.

DOWNLOAD LAYOUT: the main figure/data/code ZIP and the separate442-mask ZIP
share the Two_class_key_figures root directory. Extract both into the same
parent to restore all files. Main package alone can redraw all three figures
and reproduce the26 internal confusion matrices.
