# U-Net + ConvLSTM + GAT completed comparison

V74 completed100 real epochs, validation selected epoch45. Original V66 predictions
and optimized structured predictions are evaluated against exactly the same latest20
reference pixels and frozen binary supports. Historical V66 curves were trained on18
and are preserved separately; this file does not relabel their historical losses.

Current conditional Dice (persistent/new/tip): original .9711/.8229/.0276;
ConvLSTM+GAT root-state output .9270/.7154/.5757. Macro .6072 -> .7394.
Tip recognition improved, while new/persistent Dice declined. This is not an all-class
improvement. New labels are revised/model-assisted, and all data are one B3 field;
scores are internal consistency, not independent generalization. Same-label controls
are retained in the full experiment; a before/after comparison alone does not isolate
the effect of the two added modules from the updated supervision.

Figures show native original/reference/optimized pixels, white isolated-root background,
the same reference-selected ROI across dates and extra true-extension windows. ROI
views are illustrative and may include development pixels. No label/shape edit, color
majority voting, synthetic error or invented epoch is used. Flat auxiliary pixel scores
are separately labeled in CSV and are not mixed into the final structured predictions.

## Same-reference control

The completed100-epoch reference20 control, with the same supervision and structured
decoder but without ConvLSTM/GAT, gives persistent/new/tip Dice .9682/.8503/.4129,
macro .7438. The joint model gives .9270/.7154/.5757, macro .7394. Thus the joint
modules increase tip Dice by .1627 but reduce new/persistent Dice, and do not improve
the macro score over this matched control. No individual ConvLSTM-only or GAT-only
arm was run, so the individual module contribution is not established.
