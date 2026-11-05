# V66 U-Net versus V90: radial performance comparison

This figure restyles the previously verified pooled comparison table. Panel a shows V66 U-Net; panel b shows the final V90 reviewed workflow. Blue, orange and rose consistently identify newly observed, extending and persistent roots. Each class has Precision, Recall, Dice and IoU spokes. Pale tracks indicate 100%; radial bar LENGTH above the inner baseline is linear from 0 to 100%. The center contains the three pooled class Dice scores. These are aggregate scores, with no error bars, synthetic replicates or significance tests.

## Evaluation scope

Scores pool T2–T4 against common frozen reference B, restricted to 468,203 typed reference-root pixels. True-background pixels and the T1 baseline are excluded. Therefore these metrics describe root-type classification agreement within the evaluated root region, rather than full-image foreground/background segmentation performance. They reproduce the prior comparison TABLE; they are not medians from the preceding six-window boxplots.

V90 includes learned prediction, structural decoding and explicit manual review. The reference includes U-Net-assisted outlines, and the evaluation uses development positions. This comparison is not an independent cross-tube validation or a controlled architecture-only ablation. The frozen archive commit is recorded in source_comparison.json; it is 4615bcbd37442a3b11e5ad4533242df95f237fb6 in Annalee811/Temporal-segmentation-model. No models were retrained and no source images or labels were modified for this figure.

## Files and typography

- JPG, PNG and LZW-compressed TIFF: 600 dpi, white background.
- PDF and SVG: vector artwork with editable text; PDF embeds Times New Roman subsets.
- draw_radial.py: complete rendering script.
- V66_V90_radial_data.csv and source_comparison.json: exact plotted values and archived comparison.
- figure_caption.txt: suggested manuscript caption.
- verification.json: data, font, text-color, page-boundary and image checks.

Figure width is 190.5 mm. At this size, body labels are 8–8.5 pt and panel/model titles 11 pt. All lettering is black Times New Roman; bold is reserved for panel/model and inset titles. Check the target journal's column width before rescaling.

## Reproduce

Install numpy, matplotlib and Pillow using the versions in requirements.txt. Install a licensed Times New Roman regular/bold font locally, or provide their directory:

```bash
python draw_radial.py --fonts-dir /path/to/fonts --output-dir ./rendered
```

Expected optional font filenames are `Times New Roman.ttf` and `Times New Roman Bold.ttf`. Font binaries are not included. With Times New Roman already installed, omit --fonts-dir. The default source is source_comparison.json alongside the script. To use a different validated table, pass --source; the schema must match the bundled source.

The script does not train models, run inference, or compute new biological observations. Scores are rounded to one decimal only for display; the CSV retains full precision.
