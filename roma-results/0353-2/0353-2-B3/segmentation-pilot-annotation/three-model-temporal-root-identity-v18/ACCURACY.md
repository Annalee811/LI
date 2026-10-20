# V17 baseline and manual-continuity update

The first table uses identical held-out crops from saved full-field maps and the new unchanged frozen V17 reference. The original human audit and selectively corrected reference are reported separately. Historical V14 scores remain unchanged in the preceding archive.

|Model|Version|Precision|Recall|Dice|IoU|
|---|---|---:|---:|---:|---:|
|U-Net|V17|0.6919|0.7056|0.6872|0.6139|
|U-Net|root_identity_v18|0.6767|0.6430|0.6585|0.6023|
|DeepLabV3+|V17|0.5079|0.5013|0.5042|0.3835|
|DeepLabV3+|root_identity_v18|0.6054|0.5994|0.6021|0.4517|
|SegFormer-B0|V17|0.5434|0.6136|0.5660|0.4210|
|SegFormer-B0|root_identity_v18|0.5592|0.5999|0.5733|0.4246|

Root continuity is evaluated on common putative axes, excluding shared nodes owned by another crossing axis. Zero mixed NEW/GROWTH axes is an enforced grammar constraint, not independent biological identity accuracy. Junction-neighborhood Dice measures temporal class agreement, not instance routing correctness. Same-position development data only; external independent test images: 0.
