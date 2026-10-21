# Four-period U-Net-aligned temporal proposal and three-model continuation

Original source annotations are preserved byte-for-byte. The full upper-frame proposal follows the fixed V19 U-Net root mask and structured temporal classes, except weak ambiguous junction pixels kept temporally untyped. No artificial label error or random perturbation was introduced.

Three previously trained binary detectors stay frozen. Each branch continues its paired 31-channel temporal head in PyTorch for 16 epochs. This is not end-to-end retraining of three backbone architectures. Geometry and RGB-only features plus the train-only V19 routers are reused unchanged.

150 training / 15 validation / 18 test adjacent-date pairs (50 / 5 / 6 spatial views). The same position and dates retain the same spatial partition. Train views include overlapping shifted views; these are not independent plants or tubes. T1 baseline is displayed but excluded from the three-class temporal metric average.

Checkpoint and continuity-strength selection use validation agreement with the U-Net-derived proposal. This measures teacher consistency, not independent segmentation or biological accuracy. Original annotations are audited separately; neither set supplies an external independent test. External independent images: 0. Historical same-field exposure applies.

|Model|Internal test teacher consistency Dice|Internal test original-label agreement Dice|Selected epoch|
|---|---:|---:|---:|
|U-Net|0.9792|0.6650|8|
|DeepLabV3+|0.5768|0.4889|10|
|SegFormer-B0|0.4670|0.5017|8|

Full Precision/Recall/Dice/IoU: three_model_metrics_summary.csv and three_model_class_metrics.csv. Before/after values use identical references and spatial windows. Figures show all dates at identical native aspect ratio and one fixed validation crop. Full-frame visualizations include training regions. Single categorical ownership is enforced in each raster; candidate-root grammar restricts NEW vs PERSISTENT-with-terminal-EXTENSION. Putative axes and ambiguous crossings do not establish uniquely verified biological root identities.

The standalone annotation_review/annotate.html embeds four RGB frames and the proposal; review, undo, 60x zoom and JSON saving remain available. Source provenance and original annotation are retained.
