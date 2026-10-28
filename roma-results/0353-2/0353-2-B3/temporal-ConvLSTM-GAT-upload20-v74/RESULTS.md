# ConvLSTM + GAT temporal optimization against latest20

All six current-label modules completed100 real epochs: three joint ConvLSTM+GAT
branches and three matched controls without these modules. New temporal, memory
and graph weights started randomly. Frozen original CNN/root-mask predictions remain
inputs: this is initialization of new temporal modules, not full backbone retraining.
The old baseline was trained on18; it is reevaluated on20 here and its actual historical
figures and100-epoch logs are preserved under baseline_original, not retrospectively
rewritten. No input label, root width, length, binary contour or original weight changed.

## Conditional temporal Dice

| Model | Variant | Persistent | New | Extension | Macro |
|---|---|---:|---:|---:|---:|
| U-Net | Original | 0.9711 | 0.8229 | 0.0276 | 0.6072 |
| U-Net | Current-label control | 0.9682 | 0.8503 | 0.4129 | 0.7438 |
| U-Net | ConvLSTM + GAT | 0.9270 | 0.7154 | 0.5757 | 0.7394 |
| DeepLabV3+ | Original | 0.9638 | 0.7611 | 0.0776 | 0.6008 |
| DeepLabV3+ | Current-label control | 0.9529 | 0.7803 | 0.2985 | 0.6772 |
| DeepLabV3+ | ConvLSTM + GAT | 0.9698 | 0.8064 | 0.2652 | 0.6805 |
| SegFormer-B0 | Original | 0.9582 | 0.7788 | 0.0017 | 0.5796 |
| SegFormer-B0 | Current-label control | 0.9397 | 0.7532 | 0.0625 | 0.5851 |
| SegFormer-B0 | ConvLSTM + GAT | 0.9341 | 0.7423 | 0.1746 | 0.6170 |

The U-Net joint model improves extension recognition versus both the original and
matched control, but reduces new/persistent Dice and does not exceed the matched
control macro score. Report all classes; do not combine flat auxiliary scores with
structured final scores. No separate ConvLSTM-only or GAT-only arm was run, so their
individual causal contribution is not established. Other model outcomes are retained
in the table whether they improve or decline.

## Scope and reading the outputs

Latest20 labels include prior model-assisted revision and are not independently
reviewed across all pixels. All four dates are from one B3 position, assigned to one
outer partition. Fixed internal spatial masks provide development comparisons only,
not independent-tube/generalization accuracy. Conditional metrics score temporal
classes on the existing segmented roots. including_binary_misses additionally counts
reference roots missed by the frozen binary segmentation. The two scopes must not
be mixed. T1 is baseline, not a trained change class.

Graph targets come only from known user-confirmed identities, excluding unowned
new pixels and stale ownership; temporal class targets use20 throughout. Predicted
connections never receive manual IDs. Causal ConvLSTM/GAT preflight passed: future
T4 perturbations do not change earlier representations, and both modules have real
training gradients. Native skeleton chunks reconnect via learned affinities, endpoint
one-to-one matching and a junction-safe decoder. Predicted groups can still merge
wrong identities or fragment a true root: Root_temporal_consistency.csv reports these
errors on known heldout root labels, not just the decoder's group-level color rule.
A new-state group is uniformly new; an established group may have persistent shaft
and growing tip. This biological decoder is explicit; it is not evidence that all learned
connections are correct. Same-date and cross-date connection metrics are separate
from pixel Dice and appear in their own CSVs.

All optimized and control100-epoch loss curves use comparable train/validation
weighted pixel CE, with total multi-objective training loss separately retained in CSV.
Raw curves are not smoothed or generated. Checkpoints, optimizer/RNG resume states,
selected epochs and validation-calibrated thresholds are preserved. The selected
checkpoint may precede epoch100; all100 actual epochs remain in history.

All isolated roots have white backgrounds. Categorical panels use nearest-neighbor
resizing without root widening; black bold Times New Roman is used throughout.
Matched windows maximize reference class coverage, extension windows use reference
extension components, and high-difference windows use prediction disagreement alone,
not favorable accuracy. These illustrative windows can include development pixels.
Gains and regressions are retained; full arrays and tables are the quantitative record.
