# V35 third local field: binary segmentation with original human annotation

Twenty panels form a four-period × five-column figure: Original RGB, Human annotation, U-Net, DeepLabV3+ and SegFormer-B0. T1–T4 use identical coordinates x=1700:2120, y=300:620 from 0353-2-B3, giving 420×320-pixel crops. No whole-field images appear in the figure or bundle.

The first column is the original RGB crop. In the other four columns, existing root foreground is displayed black on white. These are direct binary segmentation masks, not color markings on RGB. Only display polarity changes; no mask pixels are added, removed, smoothed, expanded or re-annotated. Scale bars are 100 source pixels, without an assumed physical calibration. This new upper-right local field was selected for visible root morphology across the four dates, without ranking model improvement. Only the crop location changes; the underlying human and model masks are unchanged.

The human column uses the preserved original masks at results/B3_annotation_refinement_v1/data/upper/T1–T4_mask.png. Their hashes match data/summary.json, which explicitly records whole_field_original_human_masks_unchanged=true and links them to original_human_progress.json SHA256 27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09. It does not substitute the U-Net-assisted revised masks from revised_report/data/upper. Original human strokes can incompletely cover anatomical root width; this display does not correct that limitation.

Three model columns use the archived V35 binary predictions (stored as frozen binary inputs for the later temporal pipeline); they are not V66/V90 temporal classes. Their original archive provenance and source hashes are retained. No new training, inference, registration, annotations or accuracy calculations are performed. Colors do not encode root identities; all mask foreground is black.

White canvas, black Times New Roman regular/bold, 8.5–11.5 pt, 190.5×134 mm. JPG/PNG/LZW TIFF are exported at 600 dpi; PDF text and scale bars are vector and image panels are raster. The bundle includes the final exports, four original RGB crops, four original human-mask crops, 12 original model-mask crops, 16 direct black-on-white results, full scripts, caption, provenance and checksums. Font binaries are excluded.

```bash
python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py
```

Provide licensed Times New Roman.ttf and Times New Roman Bold.ttf locally. Verification checks original hashes where available, all crop arrays, identical four-date coordinates, exact mask geometry, display polarity, fonts, black text, page bounds and export dimensions. Prior figures and all original project files remain untouched.
