# Layout/color refinement of the uploaded PDF

This is a formatting-only revision of the user's uploaded V66_V90_performance_same_field_T1-T4_SCI.pdf. It does not replace the uploaded values with numbers from an earlier experiment. The output preserves every text/numeric character and its native font face, point size and weight, apart from removing the orphaned trailing vertical separator after the Macro Dice line and changing whitespace. Source point sizes remain 8.5, 9, 9.5, 10, 10.5 and 12 pt, in black Times New Roman regular/bold.

The script extracts the exact embedded TrueType subsets from the uploaded PDF and uses those fonts for the new composition. Separate font binaries are excluded from this bundle. The original PDF is included unchanged as the reproducible input and contains its original embedded font subsets.

Layout changes tighten the gap between performance plots and the four-date image matrix, align metric and image panels, regularize margins/gutters and center the Macro Dice band. The new page is 540×650 points (190.5×229.31 mm), compared with the uploaded 190.5×244 mm page. The four-period field and all 16 image panels are retained without extra crops or new geometry. The palette uses coral/clay #D38B73 for newly emerged roots, deep jade #408F83 for elongating roots and slate blue #56709C for persistent roots. Original RGB, baseline, ignored-reference pixels and all other image colors remain unchanged; only existing class-color pixels are replaced. No dilation, smoothing, registration, inference or new labels are introduced.

Important scientific note: the uploaded file labels elongating-root Dice as 75.51% and IoU as 77.5%. These are inconsistent with the standard relationship for Dice and IoU of the same class/evaluation region. The values, and the uploaded Macro Dice 88.18%, are preserved because the request is formatting-only. They are not independently validated model-performance results and should be checked before publication. The file should be treated as a layout draft until that check is completed.

JPG/PNG/LZW TIFF are exported at 600 dpi; the PDF retains vector text/charts and raster image panels. The package contains the outputs, unchanged input PDF, composition/verification scripts, extracted text-style information, image provenance and checksums. Derived image PNGs and extracted font files are recreated by the composition script and are not separately bundled.

```bash
python refine_layout.py
python verify_refinement.py
```

The source PDF hash, all 16 source image arrays, image background/geometry, output colors, every text/numeric character's font/size/weight, PDF boundaries and raster exports are checked. No earlier source files or experiment results are changed.
