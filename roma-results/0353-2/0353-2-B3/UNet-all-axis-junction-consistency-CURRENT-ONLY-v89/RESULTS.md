# Every inferred mixed axis and junction event: V89

This is an inference/structured-decoding correction on the existing U-Net + ConvLSTM/GAT result, not additional training. V88 omitted206 low/unknown-state mixed axes from its confident-state invariant, including58 axes containing both new and extension. All206 are now explicitly processed rather than omitted. The learned state head, current predicted shaft evidence and multi-period correspondence determine a MAP new/established state.59 mixed axes retain low decision-margin confidence and are exported separately; these scores are not calibrated accuracy probabilities. Root identity can still be wrong.

Every inferred new axis is new only. Every inferred older axis permits persistent shaft and/or extension but no second new birth. Current foreground, native root widths/lengths and T1 baseline remain bitwise unchanged. The previous V87 manual reference and every historical model/result file are retained unchanged. Manual classes and root IDs are not prediction inputs. No target Dice or synthetic error is used.

All shared-core footprints are grouped into junction events. The neighborhood spans two local native root diameters (radius4..10px), so outside-edge halos are checked too. New + persistent incident roots prohibit extension in the shared footprint; new + extension prohibit a persistent third type. Same-type crossings retain that type. If a forbidden extension pixel was owned by the growing neighbor but belongs to a persistent crossing strand, ownership is reassigned to the appropriate incident persistent root, rather than painting a purple island into the growing root. Tiny disconnected pieces can join a uniquely nearest new axis only with learned new-state evidence, gap<=8px and a distance margin over older axes; foreground gaps are not filled. This changes local instance ownership but never the combined foreground mask. Complex overlapping events follow the explicitly requested shared-footprint rule; biological identities at ambiguous contacts remain uncertain.

A separate scalar-color implementation verifies every non-junction unbranched segment and every junction footprint after decoding. These are consistency checks on inferred topology, not independent biological accuracy. The original spatial audit region is a known development region. Conditional temporal Dice excludes binary root misses and T1 baseline, and the same unchanged V87 reference is used before and after.

The three requested coordinate examples show mixed old V85 classes but already different V88 data. The earlier four-period comparison included OLD V85 and manual-reference columns. To remove this display ambiguity, the MAIN figures now contain ONLY current V89 predictions. Old versions appear exclusively in a figure explicitly titled DIAGNOSTIC VERSION CHECK.

Consistency summary:

```json
{
  "passed": true,
  "T1_baseline_axes_checked": 577,
  "total_four_period_axes_checked": 2960,
  "all_inferred_temporal_axes_scanned": 2383,
  "mixed_axes_before": 206,
  "mixed_new_extension_before": 58,
  "mixed_axes_after": 0,
  "new_extension_after": 0,
  "new_persistent_after": 0,
  "all_junction_events_scanned": 853,
  "forbidden_event_pixels_before": 1942,
  "forbidden_event_pixels_after": 0,
  "ownership_reassigned_junction_pixels": 479,
  "disconnected_new_root_fragments_rejoined": 6,
  "junction_radius_policy": "two local native diameters, clamped to4..10px; foreground itself unchanged",
  "low_confidence_mixed_decisions": 59,
  "geometry_unchanged": true,
  "reference_unchanged": true,
  "no_training": true,
  "scope": "Consistency of inferred axes and event footprints, NOT independent biological accuracy; MAP decisions on low-confidence axes are explicitly exported."
}
```

| Scope | Variant | Persistent | New | Extension | Macro |
|---|---|---:|---:|---:|---:|
| original_inner_spatial_audit | previous_V88 | 0.975956 | 0.837720 | 0.379040 | 0.730905 |
| original_inner_spatial_audit | current_V89 | 0.976463 | 0.846642 | 0.340480 | 0.721195 |
| all_valid_annotated_root_pixels | previous_V88 | 0.982326 | 0.887120 | 0.494603 | 0.788016 |
| all_valid_annotated_root_pixels | current_V89 | 0.981722 | 0.892259 | 0.498443 | 0.790808 |

Material metric trade-off: under the same V87 reference, the original inner spatial audit macro Dice declines from0.730905 to0.721195, while the full valid annotated-area macro rises from0.788016 to0.790808. Persistent Dice in the full area falls slightly. These consistency corrections are not claimed to improve every accuracy metric. Low-confidence counts refer to the206 mixed-axis decisions only, not to an independent verification of every other root identity.

Downloads:

- [CURRENT ONLY four-period main figure](figures/Fig1_CURRENT_ONLY_four_period_predictions_V89.tif)
- [CURRENT ONLY three requested locations](figures/Fig2_CURRENT_ONLY_requested_locations_V89.tif)
- [Complete every-axis audit](Every_axis_state_audit.csv)
- [Complete junction audit](Every_junction_event_audit.csv)
- [Low-confidence mixed-axis decisions](Low_confidence_state_decisions.csv)
- [Same-reference Dice](Temporal_Dice_same_reference_V87.csv)
