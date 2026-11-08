# Revised three-panel temporal-root percentage figure

This updates the user's previous five-metric chart by removing the skeleton-root-length curve, its axis and its legend item. The remaining four percentage means are exactly unchanged. There is now one shared 0–100% y-axis scale per panel, with three material panels (0353, A017, Q1) and T1–T4 along the x axis.

The wider plot areas (49.5 mm each), larger black Times New Roman text (10–12.5 pt), 1.9 pt lines, 6 pt markers and single bottom legend provide a more spacious publication layout. Figure size is 190.5 × 100 mm. Raster images are 600 dpi; PDF and SVG are vector artwork. Sample counts and technical notes are provided in the caption/data rather than crowded beneath each plot. Font files are not redistributed.

Plotted means come from the original Full_precision sheet of the pinned workbook, SHA256 ea74bf1853766e51fa3b30546d6893032e57e50ea991514c2735182a6fdb0e52, at Annalee811/LI commit 8b4fdb81b5938720e263d802426745c982c4479c. Each of the 36 numerical percentage means is verified against the preceding figure's values. The raw workbook and all past results remain unchanged.

Persistent, New and Extension are current-class length/current total length ×100%. E/Lprev uses extension-class length/previous total length ×100%; it is not a time-normalized growth rate. T1 is a baseline with undefined percentages, so no zero or connecting line is invented there. No new uncertainty bars or tests were introduced.

Means use valid adjacent pairs: n=6,6,6 for 0353; n=6,4,4 for A017; n=3,2,3 for Q1 across T2,T3,T4. Available-field sets can change between periods. The source includes 18 observation positions in 8 source tubes and retains registration-review flags, including Q1-1-Z2's unresolved coarse T2–T3 link. These are the original V35/V66 U-Net estimates, not V90 estimates or a new experiment.

Files include the final JPG, PNG, LZW TIFF, PDF and editable-text SVG; the plotted means and original source data in Excel; CSV/JSON values; the source workbook; a bilingual figure caption; plotting code; provenance and verification metadata.

To reproduce, install requirements.txt and provide licensed Times New Roman.ttf and Times New Roman Bold.ttf files:

```bash
python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```
