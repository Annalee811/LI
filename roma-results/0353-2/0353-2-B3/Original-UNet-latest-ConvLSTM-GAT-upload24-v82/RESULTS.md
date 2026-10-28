# Baseline correction: original U-Net plus the latest learned local ConvLSTM/GAT

The immutable baseline is the ORIGINAL U-Net shown in V75 BEFORE, not the
V75 AFTER ConvLSTM/GAT prediction and not the newer V78 U-Net. Its checkpoint
is V66 selected37, SHA25658a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382.
Its newly-observed Dice was .8229386284 against reference20. The same unchanged
prediction scores .7861928012 against the revised24 reference. These are different
reference evaluations, not model changes. The earlier .9233386899 belongs to the
PERSISTENT class of the different V75 AFTER baseline under24, not newly observed.
That previous baseline choice was mistaken for the user's intended comparison;
its files remain historical and are not relabeled as the original U-Net.

Only this original U-Net is refined. Latest V81 source24-learned local ConvLSTM/GAT
selected13, SHA256ce29d142a03c5b4ee0bf0ca9ace14e033e5bada5e1b548c8fa24ea57769646bc,
is inherited in full and locally adapted for100 additional actual epochs. The
original U-Net model, binary masks, widths/lengths, T1 and every noncandidate class
stay unchanged. DeepLabV3+ and SegFormer are not trained or changed.

Allowed edits are existing junction extension pixels, short inconsistent color
islands, history-supported persistent bridges, and small three-arm new-lateral
attachment cores. Existing continuous axes are persistent; new lateral attachments
are newly observed. A visual crossing alone is not growth evidence. Long terminal
extension and major body segments are protected. No root-wide recoloring or global
pixel class optimization is performed. Low-confidence candidates retain the original.

All supervision uses source24, preserved byte-for-byte. Separately recorded weak
history/attachment evidence supplements local training where single-period labels
may be incomplete. These proposals are not independently confirmed human labels.
Earlier same-root presence is required for persistence; future visibility alone
cannot prove earlier persistence. Manual IDs are targets only, never prediction inputs.

Historical original100-epoch curves use their actual original18 training reference;
new local100-epoch curves use24. Both BEFORE and AFTER metrics in this comparison
use the same24 reference, frozen root scope and internal spatial holdout pixels.
All dates are one B3 development position; no independent-tube accuracy is claimed.
Validation calibrates edits; test follows selection without threshold tuning.
Illustrations are selected by edit counts, not favorable accuracy. Harms, unresolved
identity conflicts, exact pixel-change audits, checkpoints and resume state are retained.

Heldout local corrections: {"changed": 893, "fixed": 714, "harmed": 32, "changed_but_still_wrong": 147, "net_corrected": 682, "edit_precision": 0.799552071668533}

Whole-field temporal Dice:

| Variant | Persistent | New | Extension | Macro |
|---|---:|---:|---:|---:|
| original_U_Net_V75_before | 0.9687 | 0.7862 | 0.0857 | 0.6135 |
| original_U_Net_plus_latest_local_CG | 0.9711 | 0.7970 | 0.0949 | 0.6210 |

Paired historical-reference audit (same original prediction and same fixed spatial/root scope):

| Reference | Original new-root Dice | Corrected new-root Dice |
|---|---:|---:|
| 20 | 0.8229 | 0.8334 |
| 24 | 0.7862 | 0.7970 |
