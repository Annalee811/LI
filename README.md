# Three joined material plots with five metrics in each plot

The final figure contains exactly three horizontally joined panels, for 0353, A017 and Q1. Every panel uses T1–T4 on the x axis and contains five mean trajectories: Skeleton length, Persistent, New, Extension, and E / Lprev.

Black circles use the LEFT axis for skeleton length in 10³ native registered pixels. The four colored percentage curves use the RIGHT axis. The left scale is 0–65 and the right scale is 0–100 in all panels, with no axis breaks, smoothing, normalization or implicit mixing of units. T1 temporal percentages remain undefined and are not plotted as zero.

The figure is 190.5 × 101 mm. All text is black Times New Roman, 8–11 pt. JPG, PNG and LZW TIFF are 600 dpi; PDF and SVG retain vector text and lines. Font binaries are not included.

## Which values are plotted

This follows the final requested layout of one plot per material with all five metrics together. Lines are the original available-field means from the user-linked workbook's Full_precision sheet. They are not six individual-sequence lines per metric, and no mean of rounded display-table values is taken. All 48 plotted numerical means were independently recomputed from the individual records only to verify equality with the frozen summary. The original workbook remains unchanged.

The linked workbook is pinned at Annalee811/LI commit 8b4fdb81b5938720e263d802426745c982c4479c, with SHA256 ea74bf1853766e51fa3b30546d6893032e57e50ea991514c2735182a6fdb0e52. It has 18 sequences within 8 source tubes, 63 available root-length records and 40 valid adjacent pairs. The five metrics reuse the V35 binary / V66 temporal U-Net trait outputs. No new training, inference, registration or V90 replacement was performed.

Root-length means use all available length records in a material/period. Temporal means use valid adjacent pairs only. Thus their sample supports can differ. The plot lists nL/nI for T1–T4 under each panel. In particular, the Q1 mean curve changes its set of available fields across periods and should not be interpreted as a fixed cohort's growth trajectory. Native pixel lengths are not calibrated millimetres or daily growth rates, and field area/sampling scales differ. Existing registration-review flags are preserved in the underlying source records, including Q1-1-Z2's unresolved T2–T3 coarse correspondence.

The three current-class percentage means have the current total length as denominator; E/Lprev uses the previous total length. Means are shown without error bars or inferential tests. Source SD remains in the Excel Source_summary sheet for traceability; 18 fields are not treated as independent tube-level replicates.

## Outputs

- Three_series_five_metric_line_charts.jpg / .png / .pdf / .svg / .tiff: the final combined three-panel figure.
- Three_series_five_metric_plot_data.xlsx: plotted means, all 72 original period cells and full-precision source summary including SD.
- plotted_means_long.csv / plotted_lines.json: exact plotted data and axis assignments.
- source_summary_full_precision.json: immutable extracted summary values.
- figure_caption_CN_EN.txt: English manuscript caption and Chinese explanation.
- plot_metadata.json / verification.json: provenance, definitions, checks and export metadata.
- sources/: original user-linked source workbook and download hashes.
- provenance/: identity policy and helper used for canonical IDs.
- make_five_metric_lines.py: complete plotting and source-value verification code.

## Reproduce

Install dependencies listed in requirements.txt and supply licensed regular/bold Times New Roman files:

```bash
python make_five_metric_lines.py --fonts-dir /path/to/fonts
python verify_figure.py
```

Expected filenames are Times New Roman.ttf and Times New Roman Bold.ttf. The source workbook is included in the package, so plotting does not require model weights or fresh image analysis.
