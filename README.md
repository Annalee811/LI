# V66 U-Net versus V90 final reviewed workflow

Comparison on the same frozen reference B: T2–T4, 468,203 typed root pixels. Archived class maps were SHA-256 checked and evaluated directly, without training or new inference. Source snapshot: Temporal-segmentation-model commit 4615bcbd37442a3b11e5ad4533242df95f237fb6.

Word: manuscript three-line table. Excel: main class metrics, Dice deltas, overall accuracy and confusion counts. ZIP: all tables and exact source/evaluation records.

V90 includes structural decoding and explicit manual review; its difference from V66 does not isolate an architecture effect. The reference includes U-Net-assisted foreground outlines. These are within-development-position reference-agreement scores, not independent cross-tube validation. True background and T1 baseline are excluded.
