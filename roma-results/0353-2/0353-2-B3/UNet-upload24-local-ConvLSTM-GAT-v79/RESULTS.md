# Source24 U-Net retraining and local-only ConvLSTM/GAT correction

Stage1 trains a new randomly initialized U-Net temporal CNN head for100 actual
epochs, using the user-uploaded24 temporal annotation. Binary U-Net masks, weights,
widths and root lengths remain frozen; the old temporal CNN weights are not loaded.
The prediction-only25-channel RGB/binary cache is reused exactly; it contains no
old temporal model output.

Stage2 freezes the newly trained U-Net temporal maps and trains a fresh causal
ConvLSTM/GAT local corrector for100 real epochs. Its pixel classification loss uses
only candidate junction/short-island pixels, with auxiliary same-root link, confirmed
parent attachment and clean-root state targets from24. No global pixel task is
optimized. Candidate masks use only frozen U-Net predictions and geometry, never
reference classes or IDs. The inherited global pixel head is frozen and unused.

Allowed edits are existing extension-colored pixels within4px of a predicted
skeleton junction, and very short green/yellow color islands (at most16 skeleton
pixels, at most72 root pixels) with same-native-path opposing-color evidence, short internal
extension/persistent bridge anomalies with earlier root support, and2px three-arm
attachment cores near newly observed lateral colors. Long
terminal extensions and major segments are not candidates. A visual crossing alone
is not evidence of a true branch. Local learned class/edit scores decide whether
to change a candidate; proposed extension at connection candidates is not applied.
The original label remains when confidence is insufficient. No forced root-wide
majority coloring or global new/established relabeling is performed.

History-supported persistent bridges and confirmed-parent new-lateral birth points
provide separately recorded weak local supervision. Earlier same-root presence is
required for persistence; future visibility alone cannot establish earlier persistence.
A newly appearing lateral attachment is supervised as newly observed, rather than
copying its persistent mother's class. Conflicting single-date labels are downweighted
only for these high-evidence local targets, while source24 remains byte-preserved.
These derived targets/proposals are not presented as independent human ground truth;
Temporal_evidence_proposals.csv records them. No direct reference-ID input is used
at prediction. Primary accuracy metrics continue to use the unchanged manual labels.

The conservative edit threshold is chosen only on the fixed inner validation mask
to reward corrected errors and penalize damage to already-correct labels. All test
scoring follows selection. Every noncandidate pixel is asserted bitwise unchanged
from the newly trained U-Net; T1 and every binary mask are also identical.

All four dates are one B3 development position. Revised/model-assisted annotation
metadata is retained; known ownership/class ambiguities are not silently edited.
Scores are internal reference agreement, not independent-tube generalization. Local
edit precision, whole-field three-class Dice and root identity conflicts are distinct
quantities. Remaining errors outside the allowed region are reported rather than
changed. Illustrations are selected by edit counts, not favorable accuracy.

Heldout local corrections: {"changed": 274, "fixed": 185, "harmed": 40, "changed_but_still_wrong": 49, "net_corrected": 145, "edit_precision": 0.6751824817518248}

Whole-field temporal Dice:

| Variant | Persistent | New | Extension | Macro |
|---|---:|---:|---:|---:|
| old_U_Net_V66 | 0.9687 | 0.7862 | 0.0857 | 0.6135 |
| new_U_Net_trained_on24 | 0.9729 | 0.8057 | 0.0470 | 0.6085 |
| new_U_Net_plus_local_ConvLSTM_GAT | 0.9732 | 0.8090 | 0.0475 | 0.6099 |
