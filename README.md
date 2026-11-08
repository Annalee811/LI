# Tight layout of the supplied Original/Reviewed U-Net comparison

This figure retains the supplied A/B composition and text labels. A contains two radial metric diagrams and central class Dice bars. B contains the same 300 × 260-pixel local field across T1–T4, with Original RGB, Reference annotation, Original U-Net and Reviewed U-Net columns.

The layout is tightened: the radial centers are 86 mm apart; the image matrix has only 2 mm between columns and 1 mm between rows. Titles, macro-Dice labels, legends and the A/B transition are placed closely while avoiding label overlaps. The crop panels retain their aspect ratio. The figure is 190.5 × 246 mm with black Times New Roman text, 8–12 pt. Raster exports are 600 dpi; the PDF preserves vector charts and lettering.

The supplied green/orange/purple/blue categorical scheme and display names are used. Crop coordinates are x=1530:1830, y=590:850 in the archived 2541 × 1108 source fields. Original RGB pixels and class-label arrays are copied exactly. Colored class renderings are generated from those labels on a white background; there is no dilation, smoothing, relabeling, model inference or new registration.

Source names: Original U-Net = archived V66 U-Net class maps. Reviewed U-Net is the display label provided by the user for the archived final V90 learned/structural/manually reviewed workflow. This is a workflow comparison and should not be described as an architecture-only U-Net ablation. The display name is preserved to match the supplied figure.

Panel A reproduces the previously verified pooled metrics exactly. Evaluation covers T2–T4 against common frozen reference B over 468,203 typed reference-root pixels, excluding true background and ignored reference pixels. T1 in panel B is a baseline; it is not included in these temporal metrics. Reference outlines include U-Net assistance, and these are within-development-position reference-agreement metrics, not independent cross-tube validation. Colors indicate temporal class, not unique root-instance ownership at crossings.

All 16 source assets are verified against archive manifests at Annalee811/Temporal-segmentation-model commit 4615bcbd37442a3b11e5ad4533242df95f237fb6. Source filenames, hashes and the ROI are recorded in source_verification.json. No historical project files or prior outputs are overwritten.

The package contains JPG/PNG/PDF/LZW TIFF, exact RGB/label crops, full source maps, quantitative source tables, provenance, a caption, plotting code and verification. Font binaries are not distributed.

```bash
python make_compact_layout.py --fonts-dir /path/to/fonts
python verify_layout.py
```

Use licensed Times New Roman.ttf and Times New Roman Bold.ttf files locally. Required dependency versions are listed in requirements.txt.
