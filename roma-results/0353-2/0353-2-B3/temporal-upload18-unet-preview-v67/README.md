# U-Net upload18 temporal preview, V67

U-Net completed100 real epochs in V66; validation selected epoch37. New inference
uses that exact checkpoint. No new training, GAT, learned routing, temporal class
constraints, root dilation or binary shape/width/length editing is applied.
The source reference is uploaded root_annotation_progress (18).json, preserved.
Reference roots without temporal typing are gray. Unreviewed background is not
shown as gray roots. T1 is baseline, not a learned change target.

Fig1: full four-period RGB/reference/U-Net overlays and white-background roots.
Fig2: same384px ROI in all dates; since T4 extension is not in this window, a
separate256px T4 inset shows all48 reference extension pixels, unaltered. Native
categorical roots use nearest-neighbor display. No missing class is fabricated.
ROIs are illustrative and reference-selected, not independent held-out test images.
Fig3: actual epoch0..100 loss, class Dice and learning-rate curves; no smoothing.
All labels Times New Roman Bold, black; 600dpi TIFF originals, PNG screen previews.

The reported JSON holdout metrics use predeclared within-field spatial masks;
they do not establish external tube/position generalization. Frozen binary misses
count as FN in end-to-end scores, and conditional scores are separately saved.
Other temporal branches may still be running; these figures are U-Net only.
