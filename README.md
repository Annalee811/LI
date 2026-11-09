# Three-column root-system workflow, real scientific panels

The user's latest three-column template is retained in structure: acquisition/alignment, deep learning, and temporal classification/dynamics. Plant, minirhizotron, U-Net architecture, recurrent/graph schematics and arrows remain illustrations. The illustrative base was edited from the supplied diagram. Exact archived scientific arrays are then placed over the replaceable data fields; the final photographic and prediction panels do not rely on generated root geometry.

The backdrop is white. All re-typeset wording is actual black Times New Roman, with italic Elymus sibiricus and roman L. At the 240 mm export width, text ranges from 9 to 18 pt. PNG/JPG/LZW TIFF are 600 dpi; PDF retains vector lettering and the quantitative chart. The schematic base and data panels are raster, so the entire PDF is not a wholly vector drawing.

## Scientific sources

The original RGB and binary U-Net masks come from 0353-2-B3 and the archived V35 binary-segmentation evaluation/V66 pipeline. Standard local panels use x=1530:1830, y=590:850, 300×260 pixels. RGBs and masks correspond to the same field across T1–T4. Binary masks are displayed white on black without changing root geometry. The perspective network inputs are an illustration of image flow, with the exact source arrays clipped to the drawn image planes. The image stack repeats the T4 photograph as a common-field/output illustration; it is not an extra sampling date.

Temporal maps are the archived V90 final learned/structural/reviewed workflow outputs. The right-hand panel is T3, x=1000:1800, y=300:1050 (800×750 pixels), providing a larger real example with all three temporal classes. Its overlay copies the existing labels: persistent=1 (blue), newly emerged=2 (yellow), elongating=3 (green). Root outlines are not smoothed, expanded, or redrawn. These colors encode temporal classes, not unique root-instance ownership at crossings. A class overlay does not establish per-instance crossing correctness.

U-Net Dice=0.8788 and root-length identity R²=0.8911 are checked against the archived V35 values 0.8787985060232514 and 0.8910977158236053. They describe the archived evaluation, not a new evaluation on these illustrative crops. Macro Dice 0.7132→0.8118 reproduces V66 versus V90 on T2–T4 against the common frozen reference B, over typed reference-root pixels. V90 includes structural/manual review. This is a within-development-position workflow agreement comparison, not an independent cross-tube validation or architecture-only ablation. T1 is the baseline and is excluded from that temporal metric.

The original sample dynamics chart mixed percentages from different series. It is replaced with a consistent archived 0353-series T2–T4 chart: means across six observation positions, not B3-only. Persistent/new/extension percentages sum to 100% at each displayed period. The source values and scope are explicitly stored in composition_manifest.json. Existing rates are not re-estimated. E/Lprev has a different denominator and is not used in this stacked chart.

No training, image registration, new inference, new annotations or new accuracy analysis was run. No original files or historical results were modified.

## Reproduction and verification

The package includes all exact cropped assets, the edited illustrative base, plotting/verification scripts, quantitative source JSON, source hashes, manifest, caption and exports. Licensed font binaries are excluded. Supply local files named Times New Roman.ttf, Times New Roman Bold.ttf and Times New Roman Italic.ttf:

```bash
python compose_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

Verification checks the embedded asset arrays, label colors, archived metric values, source hashes when full-field files are locally available, white corners, image dimensions, and the actual embedded Times New Roman fonts, black text and page boundaries. Source paths and hashes are retained for traceability; local cropped assets and export checks work without the original workspace.
