# Current dataset identities

`identity_policy.json` is the canonical identity rule for new work. `0353-1-Z2` is corrected to **Q1-1-Z2**, in tube **Q1-1**. **Q1-3-Z3 is excluded**. Historical artifacts and original file bytes stay unchanged; old identifiers can occur in explicit provenance columns only.

The active manifest has **18 positions, 66 original images, 12 complete four-date positions and 6 three-date positions**, across **8 tubes**. Missing periods remain absent. New figures and future batch outputs use canonical IDs. Four dates from one location stay in one split, and new cross-tube validation must group by canonical tube.

The latest source for Q1-1-Z2 is the root-band recheck (2314×870 common-scale candidate crop); its T2/T3 coarse localization remains unverified. Keep that review flag in model/data provenance. Existing B3-trained weights do not change when a different location's identifier is corrected.

Current Torch U-Net/SegFormer and TF DeepLabV3+ native RGB entry points, plus the latest three-model temporal prediction entry point, normalize legacy position names and reject excluded inputs before inference. New model bundles include this policy and helper. The shared batch index is the entry point for future multi-location work; it does not imply that all locations have manual segmentation ground truth.

The CSV files are metadata indices, not copies of the source images. `canonical_view_relative_path` identifies the local working view generated from the immutable registered RGB sources. This TIFF-only publication does not republish the RGB collection or retrain any model.
