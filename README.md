# V66 versus V90: performance and four-period same-field RGB overlays

The single composite contains three-class performance plots (A) and four rows of local original RGB, reference overlay, V66 U-Net overlay and V90 reviewed overlay (B). Every T1–T4 row uses exactly the same archived registered coordinates: x=500:920, y=788:1108, 420×320 pixels in the 0353-2-B3 field. No complete-field/context panels are shown. The view was selected from reference morphology and class coverage around the existing tip-containing T3/P30 region; V90 improvement was not ranked to select it. ROI_candidate_counts.json records the reference-only candidate check if retained locally.

## Colors and rendering

The muted categorical palette is consistent in the three-class plots and all image overlays: newly emerged = coral #D97B66, elongating = jade #65A69A, persistent = slate blue #5B76A4. T1 baseline is white and ignored reference pixels (255) are gray #A7A7A7. The performance plots use paired slim bars without outlines: pale bars denote V66 and the more saturated bars denote V90. Dice labels and values are emphasized. The figure background is white, all text is actual black Times New Roman, and text sizes are 8.5–12 pt at 190.5×244 mm. Raster exports are 600 dpi; PDF lettering and metric plots are vector, while the image panels are raster.

RGB pixels outside existing labels are copied exactly. Overlays replace only labelled pixels with their category color. No label smoothing, dilation, extra registration, root-geometry modification or new instance assignment is performed. Scale bars are 100 source pixels; no unverified physical calibration is introduced. Colors encode temporal classes rather than unique biological-root instance identity.

## Metric scope and sources

All 24 plotted scores (three classes × four metrics × two workflows) reproduce the archived full-precision V66/V90 comparison. Confusion matrices were recomputed from the unchanged source maps against common frozen reference B over T2–T4 and checked for exact equality. Evaluation includes 468,203 typed reference-root pixels; true background and ignored reference pixels are excluded. The scores are conditional temporal-class agreement, not full-image binary segmentation accuracy. The T1 baseline appears in the visual comparison but is not added to the temporal metrics. The performance scores do not refer only to the illustrated crop.

V66 means the archived V66 U-Net temporal output. V90 means the final learned/structurally decoded/manually reviewed V90 workflow, not neural-only predictions. The frozen reference includes U-Net-assisted outlines, and this is an agreement comparison within the development position rather than an independent cross-tube validation or controlled architecture-only ablation. Pixel counts are not biological replicates. No invented replicate scores, error bars, significance tests or new experimental results are added.

Source repository: Annalee811/Temporal-segmentation-model at immutable commit 4615bcbd37442a3b11e5ad4533242df95f237fb6. The 16 RGB/reference/V66/V90 source files match the archived SHA256 entries. Source class codes remain 0 background, 1 persistent, 2 newly observed, 3 extending, 4 T1 baseline, 255 ignored reference. All historical project data, weights, code and results remain untouched; this export resides in a separate directory.

## Files and reproduction

The bundle includes JPG/PNG/PDF/LZW TIFF, exact local RGB and label crops and overlays, archived full-field sources for reproduction, full-precision source JSON and plotted-score CSV, ROI and source manifests, bilingual caption, composition/verification scripts and checksums. Full-field sources are supplied as reproducibility inputs; they do not appear as panels in the figure. Font binaries are excluded. Supply licensed Times New Roman.ttf and Times New Roman Bold.ttf locally:

```bash
python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

The verification checks source hashes, identical four-date ROI, original label arrays and RGB backgrounds, all plotted numbers, pooled confusion matrices, actual PDF fonts and black lettering, page boundaries, white background and raster dimensions.
