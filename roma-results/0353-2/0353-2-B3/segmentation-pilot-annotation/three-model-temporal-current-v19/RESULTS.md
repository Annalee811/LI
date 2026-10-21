# Current-annotation temporal continuation

The user-selected four upper entries from (6).json are the unchanged categorical reference. T1 is baseline; T2/T3/T4 classes are relative to their preceding date. Death is excluded.

The three binary root/background networks remain frozen. Their own paired temporal heads were continued in PyTorch for24epochs. DeepLabV3+ binary inference originates from TensorFlow1.12; its temporal head is not a TensorFlow backbone retraining.

|Model|Epochs|Selected epoch|Validation structured macro Dice|Test structured macro Dice|
|---|---:|---:|---:|---:|
|U-Net|24|10|0.6366|0.6689|
|DeepLabV3+|24|22|0.3972|0.4880|
|SegFormer-B0|24|10|0.3381|0.5012|

See three_model_class_metrics.csv for Precision/Recall/Dice/IoU and three_model_period_metrics.csv for each adjacent-date result. Baseline/raw-tile and structured metrics are separately named and must not be conflated.

The validation/test windows are internal, correlated same-position samples with historical same-field model exposure. External independent images:0. Current-reference agreement does not establish independent biological accuracy. No target score was forced, and no reference pixels were altered. Candidate-axis grammar excludes NEW/EXTENSION mixtures within a decoded axis; ambiguous crossings remain flagged and biological root identity is not independently verified.

The full-frame preview includes fitting regions. Local previews always use the first validation spatial window and the same coordinates across dates/models; enlargement does not add native information.

Weights and resume.pt are stored per model. Every epoch saves optimizer/scheduler/RNG/sampler states. Old runs were preserved.
