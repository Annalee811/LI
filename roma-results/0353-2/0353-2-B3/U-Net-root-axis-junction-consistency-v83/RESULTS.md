# Whole-root temporal and junction consistency

The original U-Net binary geometry and original prediction files are preserved.
The input to this correction is the previously delivered V82 local result, itself
based on the original U-Net corresponding to historical reference20 new-root Dice .8229.
Only U-Net is processed, for all four B3 periods and the full registered upper field.

The relation model learns the actual task: which arms continue the same biological
root through a three/four-arm junction. Frozen-embedding pair-head pretraining100
actual epochs precedes100 actual joint ConvLSTM+GAT epochs. Joint losses include
continuation BCE, competing-arm ranking, within/across-period same-root identity,
root new/established state and chronological state consistency. Learned identity
affinities gate GAT messages. Direct ConvLSTM and GAT training gradients are verified.
The pair-head pretraining weights, joint selected weights, raw curves and optimizer/RNG
resume checkpoint are retained. U-Net binary weights/contours are never optimized.

The decoder is explicitly part of the method: root segments, learned arm continuations,
and temporal evidence determine complete anomalous intervals. A thick X can thin into
two Y nodes with a short shared stripe; opposite-arm geometry identifies these virtual
crossings without adding/removing a single root pixel. Both stable flanks and later
same-root evidence can recover a previously occluded interval; future presence alone
never proves earlier persistence. The final refinement is offline across all four
periods, not a future-free streaming prediction. Real terminal extension is protected.
A new root cannot have an internal extension island; established roots may have a
persistent shaft and actual growing tip. At a two-root crossing, a shared pixel may
belong to an incident root's type, never a third type. New/extension crossings therefore
retain those two types; extension is not universally banned at intersections.

These are learned history/graph features PLUS structured topology decoding, not a
claim that the raw neural pixel classifier alone removed the errors. Labels/source24
remain unchanged. Manual root IDs are supervision/audit targets, never inference input.
Uncertain pairings and temporal intervals are exported for review; not all roots are
resolved. Correct body classes outside identified intervals/intersections are retained.
Native root widths, lengths and all four foreground masks remain bitwise identical.

All dates belong to one B3 development position. Source24 contains historical
model-assisted annotation metadata and is not fully independently reviewed. Pixel
metrics are internal reference agreement on frozen root supports, separately scored
against20 and24; segmentation misses are outside that conditional task. Known-root
new/extension conflicts and junction third-type checks have different denominators.
The latter uses confirmed IDs and reference arm types for scoring, not the decoder's
own proposed allowed types. The reference itself can contain inconsistent core labels,
which are recorded. These counts do not constitute independent biological accuracy.
The user-reported window is a known development/regression example; other displayed
windows use change density only and can include regressions. No target score or
synthetic error was introduced, and old results were not overwritten.

Internal joint continuation-head task: {"correct": 109, "accepted": 118, "available": 160, "accuracy": 0.923728813559322, "coverage": 0.7375}; full accepted/all task counts are in joint_training_results.json. This is root-arm pairing, not segmentation Dice.

Current structural checks:

{
  "root_period_records": 158,
  "root_new_extension_conflicts5": {
    "original_U_Net": 24,
    "previous_local_V82": 17,
    "whole_segment_temporal_graph": 11
  },
  "scoreable_confirmed_two_root_junctions": 80,
  "third_type_junction_audit": {
    "original_U_Net": {
      "junctions_with_third_type": 13,
      "third_type_pixels": 202
    },
    "previous_local_V82": {
      "junctions_with_third_type": 10,
      "third_type_pixels": 79
    },
    "whole_segment_temporal_graph": {
      "junctions_with_third_type": 6,
      "third_type_pixels": 52
    }
  },
  "holdout_edit_counts": {
    "changed_on_holdout": 1166,
    "fixed": 1037,
    "harmed": 19,
    "changed_still_wrong": 110
  },
  "user_review_case_yellow_pixels": {
    "original_U_Net": 325,
    "previous_local_V82": 241,
    "whole_segment_temporal_graph": 1
  },
  "user_case_scope": "User-reported known regression region; not independent validation",
  "reference24_itself_has_core_third_type_pixels": 0
}

Conditional temporal Dice:

| Reference | Variant | Persistent | New | Extension | Macro |
|---|---|---:|---:|---:|---:|
| 20 | original_U_Net | 0.9711 | 0.8229 | 0.0276 | 0.6072 |
| 20 | previous_local_V82 | 0.9732 | 0.8334 | 0.0352 | 0.6139 |
| 20 | whole_segment_temporal_graph | 0.9756 | 0.8617 | 0.0534 | 0.6303 |
| 24 | original_U_Net | 0.9687 | 0.7862 | 0.0857 | 0.6135 |
| 24 | previous_local_V82 | 0.9711 | 0.7970 | 0.0949 | 0.6210 |
| 24 | whole_segment_temporal_graph | 0.9736 | 0.8261 | 0.1261 | 0.6419 |
