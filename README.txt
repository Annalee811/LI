Compact A/B/C SCI figure
Created: 2026-11-23

Exactly three horizontal panels:
A - essential class-specific temporal baseline comparison;
B - paired module and structural-decoder effects in one forest plot;
C - one independent-position prediction montage, matched RGB field at T1-T4.

Output: 190 x 100 mm; Times New Roman; black text; white background.
JPG/PNG/TIFF: 600 dpi. PDF: vector plots/text plus raster RGB crops.
SVG: editable vector elements and embedded image data; install Times New Roman for consistent text rendering.

Reproduce the figure (Python with numpy and matplotlib):
python make_ABC_compact.py --fonts /path/to/fonts
The fonts directory must contain Times New Roman.ttf and Times New Roman Bold.ttf.
Font files are not redistributed. No model, training data or GPU is needed to replot.

ABC_plot_values.csv records the means, seed SDs and reference contexts actually plotted.
Four source CSVs retain all available internal comparisons, including variants omitted from the compact display.
Independent_Q1-2-B3_image_cache.npz stores unchanged original RGB crops and class masks.
Mask codes: 0 background, 1 existing, 2 new, 255 temporally unassigned baseline.
Independent_evaluation_status.json deliberately retains null independent accuracy fields.
Figure_caption_and_notes.txt provides English and Chinese captions and interpretation.
Verification.json checks resolution, fonts, layout and reuse of source inputs.
SHA256SUMS.txt records hashes for the figure package, before ZIP creation.

Existing sources were preserved. The compact figure reuses the completed Nov 23 two-class evaluation and independent image cache. Its source release is:
https://github.com/Annalee811/LI/tree/3839df9feaa49ecd30ffeec5f1d6593d205686d5
This package is a figure and source-data package; training/deployment weights are in the earlier project releases.

Important: merging class masks is not two-class retraining. Internal root-pixel evaluation is not end-to-end full-background segmentation accuracy. Panel B has two distinct reference contexts. Panel C lacks human temporal ground truth and does not provide independent accuracy.
