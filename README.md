# Compact SCI figure: local RGB overlays and radial V66/V90 comparison

The final composite is **190.5 × 201 mm** (near square), designed for a full double-column layout. All text is black Times New Roman, 8–11 pt at the exported size. JPEG, PNG and LZW TIFF are 600 dpi; the PDF retains vector labels and annotations with embedded font subsets. Font binaries are not distributed.

## Panel a: two illustrative local examples

Each row contains the original archived registered RGB field with a yellow locator box, the unchanged RGB crop, and matched reference, V66 U-Net and V90 reviewed class overlays. All crops are exactly 256 × 256 source pixels. Scale bars denote 64 pixels; no unverified physical calibration is applied. Label arrays are cropped directly without dilation, smoothing or relabeling. Only foreground class pixels are blended with color at opacity 0.85; background RGB is unchanged. Grey indicates ignored reference pixels (255).

- R1: T2, archived window P33, x=1428, y=852. This dense crossing-root network includes 5,909 persistent, 2,855 newly observed and 190 extending reference pixels, plus 42 ignored pixels.
- R2: T3, archived window P30, x=571, y=852. This tip-containing region includes 3,238 persistent, 70 newly observed and 247 extending reference pixels, plus 3 ignored pixels.

Coordinates are zero-based, from the upper-left corner. Both windows were taken from the existing V66 spatial test-window manifest. They were chosen for visible morphology and coverage of temporal categories; no optimization or ranking of V90 improvement was used. The two rows are different places and periods, not a tracked temporal sequence. They illustrate local behaviors and do not estimate population-level performance. Colors encode temporal class, not unique root-instance ownership. The images alone do not establish instance identity at crossings.

The archived registered RGBs are 2541 × 1108 pixels. Their SHA256 hashes match both the V66 RGB archive and the V90 base RGB archive. No new image registration, model training or inference was run. Original project files, maps and past figures remain unchanged.

## Panel b: unchanged pooled radial metrics

The two circular plots reproduce the exact percentages from the previously verified V66/V90 comparison table. They do not display metrics recomputed from the two illustrative crops. Precision, recall, Dice and IoU pool T2–T4 over 468,203 typed reference-root pixels against common frozen reference B. True-background pixels and the T1 baseline are excluded. Consequently these are temporal root-class agreement metrics conditional on the evaluated root region, not full-image binary segmentation scores.

The palette is consistent throughout: newly observed = blue #75BED8; extending = orange #E6AA75; persistent = rose #D7838C. Pale outer tracks indicate 100%. Radial length ABOVE the inner baseline is linear from 0 to 100%; color-sector area is not the scale. Center bars show pooled class Dice; macro Dice is the unweighted average across the three classes. No synthetic replicates, uncertainty bars or significance tests were added.

V90 denotes the final learned, structurally decoded and explicitly manually reviewed workflow. The reference includes U-Net-assisted outlines. The comparison uses development positions and should not be described as a controlled architecture-only ablation or independent cross-tube validation.

## Provenance and files

Source archive: Annalee811/Temporal-segmentation-model at immutable commit `4615bcbd37442a3b11e5ad4533242df95f237fb6`. Source maps and hashes are recorded in source_verification.json; crop coordinates and selection rationale are recorded in roi_manifest.json. Source class codes remain unchanged: 0 background, 1 persistent, 2 newly observed, 3 extending, 255 ignored reference.

- V66_V90_compact_SCI.jpg / .png / .pdf / .tiff: final composite.
- marked_source_RGB/: full-resolution locator images, with RGB context and a box marking the exact crop.
- crops/: original RGB crops, exact label crops and RGB overlays, saved separately for reuse.
- sources/: immutable RGB/map copies and archived spatial manifest used for reproduction.
- source_comparison.json and radial_metrics.csv: full-precision pooled metric values.
- figure_caption_CN_EN.txt: suggested English manuscript caption and Chinese explanation.
- make_compact_figure.py: full composition and overlay code.
- verify_figure.py and verification.json: source, pixel, metric, typography, boundary and export checks.

## Reproduce

Install dependencies listed in requirements.txt. Install a licensed Times New Roman regular and bold locally, or supply a directory containing `Times New Roman.ttf` and `Times New Roman Bold.ttf`:

```bash
python make_compact_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

When Times New Roman is already installed, omit --fonts-dir. An optional --output-dir changes the export directory; the supplied verification script checks the default output directory alongside itself. Data and class labels must not be altered when adjusting layout.
