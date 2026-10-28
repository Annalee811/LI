# Latest learned ConvLSTM/GAT applied to the user-selected V75 U-Net

Only U-Net is optimized. DeepLabV3+ and SegFormer are not trained or changed.
Two sources have deliberately different roles: the exact V75 AFTER class maps
(selected45, originally from V74) are the immutable body-class baseline; the latest
V79 source24-trained local ConvLSTM/GAT weights (selected92) are the learned
correction module. We inherit the latter's entire learned state, including local
heads, rather than reinitializing it or inheriting the older selected45 representation.

100 real additional local adaptation epochs use source24 labels/confirmed root
identities. V78/V79 U-Net output maps are NOT used as the body baseline. Pixel
priors are updated to the user-designated V75 prediction. The new module can edit
only defined small junction/short-island/attachment/bridge candidates. All other
body classes, binary masks, T1, widths and lengths stay unchanged. The original
V75 predictions, old weights and all older runs remain preserved.

Existing continuous axes are persistent. New lateral attachments are newly observed,
rather than copying the mother's persistent state. Earlier same-root evidence is
required for persistence; future visibility alone cannot prove earlier persistence.
A visual crossing alone is not a new-growth event. Long terminal extension is
protected. Source24 is byte-preserved; derived weak temporal/attachment targets
are separately recorded and never described as independent human ground truth.
Manual IDs are supervision only, not prediction inputs. Validation conservatively
selects edit confidence; test metrics follow selection without threshold tuning.

Historical base100-epoch curves were trained on20 and retain that label; the new
local100-epoch curves use24. All BEFORE/AFTER metrics in this report use exactly
the same24 reference and full-map internal holdout pixels. Historical V75 source20
Dice cannot be directly compared with the current24 Dice. All dates are one B3
development position, not independent test tubes. Illustrations use edit counts,
not favorable accuracy. Harms and unresolved same-root conflicts remain in CSVs.

Heldout local corrections: {"changed": 371, "fixed": 286, "harmed": 85, "changed_but_still_wrong": 0, "net_corrected": 201, "edit_precision": 0.77088948787062}

Whole-field temporal Dice:

| Variant | Persistent | New | Extension | Macro |
|---|---:|---:|---:|---:|
| old_U_Net_V66 | 0.9687 | 0.7862 | 0.0857 | 0.6135 |
| designated_V75_U_Net_ConvLSTM_GAT | 0.9233 | 0.6802 | 0.5175 | 0.7070 |
| designated_V75_plus_local_correction | 0.9250 | 0.6828 | 0.5131 | 0.7070 |
