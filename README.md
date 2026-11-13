# V66 versus V90: three temporal Dice plots and a new matched local field

Compact composition: 220 × 186 mm, white background, actual black Times New Roman, 9–13 pt; 600-dpi JPG/PNG/LZW TIFF and a PDF with vector text/curves and raster image panels. At 190 mm reproduced width, 9 pt becomes approximately 7.8 pt. Legends contain only the V66/V90 line styles and three temporal-class colors.

Panel A shows real per-date Dice for newly emerged, elongating and persistent roots over T2–T4, not training curves or interpolated samples. V66 uses dashed lines/open circles and V90 solid lines/filled circles. Adjacent measured points are connected directly, without smoothing. At T4 reference B has zero elongating-root pixels: the two corresponding points are n.a., not substituted with zero. Raw false-positive counts at that date remain in the supplied table and original pooled comparison. Values can decrease as well as increase: V90 does not outperform V66 at every date/class. No replicate observations, error bars or significance tests are invented.

Panel B contains one Original RGB column, one Human reference column, one V66 column and one V90 column, all at exactly the same registered coordinates for T1–T4: x=1000:1480, y=500:760 in 0353-2-B3, 480 × 260 source pixels. The ROI was selected using reference morphology and class coverage only, without ranking V90 improvement. Candidates extending outside the source images were rejected. These are local images only, with no whole-field panels. The 100-pixel scale in T1 Original RGB applies to all panels. No unverified physical calibration is introduced.

Colors: amber/orange #E69F35 newly emerged; purple/magenta #B64E8B elongating; blue #3F75C4 persistent. T1 baseline is white; ignored reference pixels are gray #808890. Only existing labelled pixels are recolored. The nonlabelled RGB background and label geometry remain exact. Colors denote temporal categories, not unique biological root-instance identities. No new dilation, smoothing, inference, registration, training or annotation is performed.

## Evaluation scope

Each date is evaluated over all typed reference-root pixels in the complete archived field, not just the illustrated ROI. True background and ignored reference pixels are excluded; missed roots remain false negatives. The common frozen reference is B. T1 is baseline and excluded from temporal scores. Per-date confusion matrices sum exactly to the original T2–T4 pooled matrices over 468,203 typed reference-root pixels. The original pooled macro Dice values remain 71.3237098784388% for V66 and 81.17740562710995% for V90 (see full-precision source JSON for exact values). A pooled Dice is not an unweighted mean of the plotted date-specific Dice values.

V66 means the archived U-Net temporal classification output. V90 means the final learned, structurally decoded and manually reviewed V90 workflow, not neural-only predictions. Frozen human reference B includes manual temporal assignments and pre-existing U-Net-assisted binary outline refinement; it is not a pure unaided human stroke mask or independent blinded anatomical ground truth. This is within-development-position agreement, not independent cross-tube validation or a controlled architecture-only ablation. See the bilingual caption for publication wording.

## Provenance and reproduction

Source repository: Annalee811/Temporal-segmentation-model, immutable commit 4615bcbd37442a3b11e5ad4533242df95f237fb6. All sixteen original RGB/reference/V66/V90 source-file hashes were checked against the archived source manifest before cropping. Twelve full-field class arrays are included as a compact NPZ for exact metric recomputation; full-field RGB photographs and font binaries are not republished in this compact bundle. All sixteen local RGB/label crops and twelve exact overlays are included. Historical project files, results and weights are preserved.

Reproduce with Python and the versions in requirements.txt. Supply licensed Times New Roman.ttf and Times New Roman Bold.ttf locally:

```bash
python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

The optional prepare_data.py can regenerate the compact inputs from the original verified source bundle:

```bash
python prepare_data.py --source-dir /path/to/V66_V90_same_field_SCI_20261113
```

Data files: temporal_Dice.csv and temporal_metrics.json contain all date-specific TP/FP/FN/support and scores, with null/empty Dice for no reference positives. source_comparison.json retains the original pooled scores. data_manifest.json and source_verification.json retain crop coordinates, source identities and SHA256 provenance. verification.json records independent recomputation, exact masks/backgrounds, fonts, text bounds and export dimensions. SHA256SUMS.txt covers all bundled files except itself.
