# V14 baseline and manual-continuity update

The first table uses identical held-out crops from saved full-field maps and the new manual-class/U-Net-morphology reference. Original published V14 teacher scores use another reference and tilewise inference, and are preserved separately.

|Model|Version|Precision|Recall|Dice|IoU|
|---|---|---:|---:|---:|---:|
|U-Net|V14|0.6688|0.6914|0.6788|0.5923|
|U-Net|manual_continuity_v17|0.6919|0.7056|0.6872|0.6139|
|DeepLabV3+|V14|0.4953|0.4964|0.4907|0.3716|
|DeepLabV3+|manual_continuity_v17|0.5079|0.5013|0.5042|0.3835|
|SegFormer-B0|V14|0.4411|0.4646|0.4496|0.3433|
|SegFormer-B0|manual_continuity_v17|0.5434|0.6136|0.5660|0.4210|

Root continuity is evaluated on common putative axes, excluding shared nodes owned by another crossing axis. Zero mixed NEW/GROWTH axes is an enforced grammar constraint, not independent biological identity accuracy. Junction-neighborhood Dice measures temporal class agreement, not instance routing correctness. Same-position development data only; external independent test images: 0.
