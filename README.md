# V35 local root segmentation on original RGB across four periods

The single figure contains 16 panels: T1–T4 rows and Original RGB, U-Net, DeepLabV3+ and SegFormer-B0 columns. Every panel uses the identical registered coordinates x=500:920, y=788:1108 (420×320 source pixels) from 0353-2-B3. No whole-field panels, metric plots, reference columns or temporal-class illustrations are added.

The masks are the original V35 binary predictions, archived as frozen binary inputs to the later V66 pipeline. They are not V66/V90 temporal-category maps. Their source hashes match the frozen resource manifest at Annalee811/Temporal-segmentation-model commit 4615bcbd37442a3b11e5ad4533242df95f237fb6. The RGB and all 12 source binary masks were checked before cropping. Source paths, stage names and SHA256 values are retained in source_provenance.json; the bundle contains only the local source crops, without whole-field image copies.

All models use the same teal #58A896 to mark existing predicted foreground. This is a binary foreground legend, not a color code for persistent/new/extending classes, uncertainty, or root-instance identity. Outside the original predicted mask, every RGB pixel remains unchanged in lossless local overlays. No thinning, dilation, smoothing, mask revision, new registration, training or inference is performed. Masks and crop arrays retain native source pixels. The qualitative field is carried over from the previous matched-view comparison; no per-model score ranking was used to choose a favorable crop. No new accuracy claims are made from this illustration.

White canvas; black Times New Roman regular/bold, 8.5–11.5 pt; 190.5×166 mm; JPG/PNG/LZW TIFF at 600 dpi. The PDF retains vector text and scale bars, with raster image panels. Scale bars mark 100 source pixels, without assuming physical calibration. Original RGB crops are separately supplied without scale-bar overlays.

The package includes exports, four exact RGB crops, 12 original binary-mask crops, 12 lossless foreground-on-RGB overlays, bilingual caption, composition/verification code, dependency versions, source provenance and checksums. Font binaries are excluded. Provide licensed Times New Roman.ttf and Times New Roman Bold.ttf locally:

```bash
python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

Verification checks source/crop hashes, same four-date coordinates, binary mask values, unchanged overlay background and geometry, the font faces, black lettering, page bounds, dimensions and export resolution. Full-source hash checks run additionally when the recorded source paths are available. Historical project files and earlier figures are not changed.
