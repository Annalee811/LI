V90 downstream two-class composition — 2026-11-23

Default groups: New = original 2; Existing = original 1 + 3. This is category
grouping after prediction, not retraining a two-output network. Root-axis/track
maps are copied byte-for-byte; root instances are not merged by this grouping.
Existing refers to root-axis identity and includes newly elongated portions.

Rates are current skeleton-length fractions, not pixel-area fractions or root counts.
The same binary skeleton is used for all groups. Non-redundant H/V edges weigh 1,
diagonals weigh sqrt(2), isolated pixels weigh 1; half each edge goes to each endpoint.
L_existing = L_persistent + L_extension; r_new + r_existing = 100% on typed frames.
This conservation is not a claim of 100% predictive accuracy.
T1 has no temporal fractions. Missing dates are NA, not zero. No mm/day conversion.

Primary figure: V90_three_series_two_class_rates_SCI.*
Alternative 3-D figure: V90_three_series_two_class_3D_SCI.*
Main data: Series_mean_adjacent.csv (43 actual adjacent comparisons).
Gap-inclusive data: Series_mean_previous_observed.csv (48 observed intervals).
Five gap comparisons are not mixed into the main adjacent-stage curves.
Tube-balanced, coarse-position-excluded and assigned-support sensitivities are separate.
Main means are per-field arithmetic means, matching the previous descriptive convention;
SD is a sample field SD, not SEM/CI or a count of independent biological replicates.
There are 18 fields nested within eight tubes. Category SDs are recomputed after grouping.
Two complementary percentages have equal SD; this is not a duplicated-data error.

Raw primary shares retain provisional class assignments for all predicted root support.
Original uncertainty flags are retained in alternate maps and in skeleton cache files.
Assigned-support shares use only unflagged support, with a different denominator;
use their coverage and keep them separate from raw primary shares.
No same-format flag is available for original B3: its assigned-support sensitivity is NA.
The Q1-1-Z2 T2/T3 coarse-location flag remains. Q1-3-Z3 is excluded.

Actual data source: original automatic V90 on B3, frozen V90 transfer adapter on other
positions. Adapter approximations remain documented; independent temporal GT is unavailable.
The old three-series workbook used V35/V66 and is not relabelled as V90 here.

Class PNG codes: 0 background, 1 existing, 2 new, 255 baseline/unassigned.
The uncertainty-retained PNG also uses 255 for original flagged root pixels.
The raw group map preserves exactly the previous new-root pixels; the uncertainty
map intentionally withholds flagged pixels rather than asserting a class.
On-white maps use blue for existing, gold for new and gray for unassigned/baseline.

V90_two_class_rates.xlsx contains all66 observed frames,72 planned cells, field/tube
means and quality sensitivities. The Word file has editable results and a three-line table.
two_class_policy.json fixes the downstream definitions for later analyses.

Portable numeric reproduction (NumPy, Pillow):
python recompute_from_skeleton.py --bundle . --output /new/empty/reproduction
Cached native skeleton support, endpoint weights and original labels reproduce all66
fractions without model inference. The saved check matched every rate within1e-8.

To group a future V90 prediction, without merging identities:
python regroup_v90_prediction.py --classes /path/Tn_classes.png  --ownership /path/Tn_ownership.npz --output /new/empty/two-class-prediction
To rebuild raw extraction, calculate_two_class_rates.py needs the source project and
its active manifest; use --output with a new directory. To redraw figures, use
make_two_class_figures.py --fonts /path/to/licensed/Times-New-Roman/fonts.
Fonts are not redistributed. All diagrams use black Times New Roman text on white.
Python dependencies: numpy, Pillow, scikit-image, matplotlib, openpyxl, python-docx.
No training, annotation editing, image registration or original-file overwrite was performed.
