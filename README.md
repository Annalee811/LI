# Two candidate roots: axis ownership and temporal segments on original RGB

A compact four-panel figure shows only two existing candidate roots at one crossing: original RGB, archived before-axis fragments, refined candidate axes, and temporal root-segment labels. R1 maps to source axis327, R2 to axis388. Frozen source versions are V17 manual continuity and V18 root identity refinement. No new predictions, registration, model training or manual labels are made. Historical project files remain unchanged.

All panels use the same 240×180-pixel ROI at x=1530:1770, y=870:1050 in T2 of 0353-2-B3. Only the common foreground domain belonging to the two final candidates is highlighted. Other RGB pixels, including nearby unselected roots, remain exact. This common-domain selection is a qualitative illustration, not an independent evaluation or unbiased improvement estimate.

Before-axis fragments come from five actual archived IDs:347,417,442,447,458. They are rendered neutral gray; open markers show actual source-path endpoints. These are fragments of the displayed candidate domain, not five independently identified biological roots. The refined raster has two exclusive integer IDs. No extra pixels are bridged, filled, smoothed or reassigned. Refined colors distinguish candidate routes; temporal colors distinguish classes, not root instances.

The source class labels on R1 are newly emerged; R2 has persistent and elongating segments. Existing uncertainty overrides confident class/identity coloring in 84 selected pixels. Both complete source paths retain unresolved-crossing flags. Candidate ownership is model-supported and is not independent biological instance ground truth. No identity-accuracy percentage, new Dice or artificial significance result is reported.

White canvas, black actual Times New Roman, bold titles/labels/legend, 210×76mm, 600dpi JPG/PNG/LZW TIFF and vector PDF lettering with raster photos. Native detail is limited to the original 240×180-pixel crop; export scaling does not invent photographic information. Full bilingual publication wording is in figure_caption_CN_EN.txt.

exact_inputs.npz stores exact compact arrays for reproduction; selected_paths.npz retains the relevant actual paths. source_checks.json lists full source paths and SHA256 hashes. data_manifest.json records array hashes, IDs, class runs, uncertainty and selection. Original_RGB_crop.png and the other three lossless panel PNGs contain unannotated image arrays; circles, scale and labels are separate vector annotations in the PDF. make_figure.py reproduces the composition; verify_figure.py checks exact arrays, existing scalar ownership, uncertainty, fonts, bounds and exports. prepare_inputs.py optionally rebuilds compact inputs from the existing workspace source files.

Provide licensed Times New Roman.ttf and Times New Roman Bold.ttf locally (font binaries excluded):

python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
