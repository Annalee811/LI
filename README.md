# Three-dimensional filled line charts with marked means

This figure follows the supplied 3D filled-line reference while using only the existing root-trait percentage means. It contains three material panels (0353, A017, Q1), four semi-transparent categorical layers per panel, dashed black lines and circular markers at the actual mean points. Each of the 36 measured means is annotated to two decimals.

The x axis is Period (T1–T4), depth is Trait, and the common vertical scale is Mean (%) from 0 to 100. Trait depth abbreviations map to the shared legend: P = Persistent, N = New, E = Extension, R = E / Lprev. The smaller layers are in front and the tallest persistent layer behind, reducing occlusion. Orthographic projection and identical camera/scale settings are used in all panels.

Only T2–T4 have defined temporal percentages. T1 remains a blank baseline. Straight line segments join actual observations, without smoothing, interpolated peak values, artificial noise or fabricated extra observations. Polygon closure to z=0 is only the geometric fill boundary, not a measured zero. Depth categories are separated, not cumulatively stacked. The previously removed total-root-length curve remains omitted.

All values are unchanged from the prior four-percentage chart. Persistent/New/Extension divide current class length by current total length; E/Lprev divides extension-class length by previous total length. Means use valid adjacent pairs: 6,6,6 for 0353; 6,4,4 for A017; 3,2,3 for Q1 across T2,T3,T4. These are descriptive field-level means, from 18 positions in 8 source tubes, without error bars or significance tests. Available-field composition can change across periods. The existing registration-review flags, including Q1-1-Z2's unverified T2–T3 coarse link, remain in the source records. Frozen V35/V66 U-Net estimates are reused; no new experiment or V90 replacement was made.

Figure size: 240 × 112 mm. Text is black Times New Roman. JPG/PNG/LZW TIFF are 600 dpi; PDF and SVG preserve vector text/graphics. Source data, values, captions and the complete plotting script are included. Raw font files are not redistributed.

```bash
python make_3D_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

The font directory must contain licensed Times New Roman.ttf and Times New Roman Bold.ttf files. Dependencies are pinned in requirements.txt. The original source workbook remains unchanged, with SHA256 ea74bf1853766e51fa3b30546d6893032e57e50ea991514c2735182a6fdb0e52 at Annalee811/LI commit 8b4fdb81b5938720e263d802426745c982c4479c.
