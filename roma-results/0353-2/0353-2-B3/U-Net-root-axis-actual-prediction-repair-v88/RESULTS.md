# Root-axis temporal consistency: prediction repair, not another reference edit

This changes the U-Net temporal prediction maps on all four registered upper fields, starting from V85. It does NOT change the user reference, root masks, widths, lengths, original CNN weights, ConvLSTM/GAT weights, or historical predictions. No new optimization epochs are claimed. Existing learned ConvLSTM/GAT identity and state features are reused with explicit chronology and axis decoding. This is learned-feature inference plus structural decoding, not a claim that neural training alone fixed the examples.

A prior root may have a short old shaft and a much longer later extension. Its new-to-old length fraction cannot create a second birth. Same-prior-root arm continuity can correct junction pairing even when the continuing mother bends more than a new lateral. A first-observed root axis can contain false persistent as well as false extension islands: current state-head evidence, absent earlier aligned roots and later same-axis visibility support a whole-axis new state. An established root can contain persistent shaft and genuine distal extension, with approximate earlier-tip boundaries recorded when uncertain. Shared pixels have one incident axis owner; a true new lateral is separate from its mother. A growing tip merely touching another root is not automatically a daughter.

The old arbitrary .55 same-root acceptance gate is replaced by the inherited validation-calibrated connection threshold. No threshold is selected for a target Dice. Previous root lengths and contours are frozen. The final refinement uses all four fields including later visibility; it is offline, not future-free streaming inference. Future visibility alone never proves earlier presence. Some short/occluded/ambiguous axes remain unresolved. All accepted new/established axes satisfy their semantic state invariant over the owned native-width pixels, including halos. This invariant is NOT independent accuracy; the ownership inference can still be wrong.

Known examples:

- A: Existing T1 root: false new-colored distal shaft becomes extension. Changed pixels on identified axis: 696.
- B: First-observed axis: false persistent / extension intervals become new. Changed pixels on identified axis: 670.
- C: Two roots contact: upper root already visible in T2; horizontal root first appears in T3. Changed pixels on identified axis: 50.
- D: Existing root: false new interval becomes growth; old shaft remains persistent. Changed pixels on identified axis: 137.

Example C is explicitly retained as two touching roots: the diagonal root is already visible in T2 and is manually numbered N-0043; the horizontal root first appears in T3 and is N-0041. Root-ID overlays are audit display ONLY, loaded after predictions were frozen. No inference reads these root IDs. Their temporal colors can legitimately meet at a contact; they must not be merged to manufacture uniformity. Example D separates the older0002 axis from the N-0042 axis first observed in T2; false new coloring on the older axis is repaired, while genuine distal growth is preserved.

The V87 reference remains byte-for-byte unchanged: earlier two T2 reference corrections and all T4 extension-to-new changes are preserved. Metrics are reference agreement under this SAME reference. All B3 is a development position; these user-reported examples include previously audited spatial areas, so old inner-test scores are now described as spatial audit, NOT independent validation. Scores are conditional on fixed-root foreground and exclude binary segmentation misses and T1 baseline.

| Scope | Prediction | Persistent | New | Extension | Macro |
|---|---|---:|---:|---:|---:|
| original_inner_spatial_audit | previous_V85 | 0.973197 | 0.858445 | 0.231687 | 0.687776 |
| original_inner_spatial_audit | root_axis_V88 | 0.975956 | 0.837720 | 0.379040 | 0.730905 |
| all_valid_annotated_root_pixels | previous_V85 | 0.978773 | 0.874175 | 0.379577 | 0.744175 |
| all_valid_annotated_root_pixels | root_axis_V88 | 0.982326 | 0.887120 | 0.494603 | 0.788016 |

Root state and historical confirmed-root checks (different denominators):

```json
{
  "confident_axis_records": 1194,
  "confident_new_axes": 172,
  "confident_established_axes": 1022,
  "forbidden_class_pixels_before": 12108,
  "forbidden_class_pixels_after": 0,
  "historical_reference_root_records": 158,
  "historical_known_root_new_extension_mixes": {
    "previous_V85": 11,
    "root_axis_V88": 11
  },
  "all_axis_paths": 2960,
  "unresolved_axis_states": 1189,
  "note": "Accepted-axis state checks are a decoder invariant, not independent accuracy. Historical human-root IDs can span different inferred axes; their separate audit is retained."
}
```

Downloads:

- [Reported examples: actual prediction repairs](figures/Fig1_User_examples_actual_prediction_repairs.tif)
- [Four-period full-field comparison](figures/Fig2_Four_period_root_axis_comparison.tif)
- [Example C: four-period numbered-root audit](figures/Fig3_Example_C_four_period_identity_audit.tif)
- [Class metrics under the same V87 reference](Temporal_class_metrics_same_V87_reference.csv)

Material limits: the former inner spatial audit new-root Dice decreases from0.858445 to0.837720, despite full annotated-area new-root Dice increasing from0.874175 to0.887120. All three full-area class Dice improve; this is not a claim that every subset improves. The158 historical confirmed-root records still contain11 mixed new/extension cases, unchanged from V85. Do not substitute the zero-forbidden-class decoder invariant for that independent-of-decoder identity audit. Those mixed records can reflect unresolved routes, original ownership conflicts and remaining errors; they are not asserted fixed. Many unclassified computational axes are short fragments, so their count is not a number of biological roots or verified errors.

[Unchanged latest user reference JSON](https://raw.githubusercontent.com/Annalee811/LI/b8c020c576cb11419d1db795da081cfd5e155fb8/roma-results/0353-2/0353-2-B3/U-Net-T4-reference-extension-to-new-v87/root_annotation_progress_24_user_corrected_T2_and_T4.json)
