# White-background workflow with real B3 image panels

The supplied workflow's four-stage composition is retained: data acquisition, alignment/segmentation, spatiotemporal classification, and dynamics/applications. Plant, soil, tube/camera, U-Net, recurrent/GAT, applications, qualitative bar chart and final root-class illustration remain schematic elements. The pale backdrop is made white, and the text is re-typeset in actual black Times New Roman. The binomial Elymus sibiricus is italic and L. remains roman.

The edited illustrative base was made from the user's supplied diagram. The publication export then places the verified scientific RGB and U-Net mask arrays over the replaceable image slots, so the final research panels do not rely on generative reconstruction of root geometry. There are 13 such panels: four top RGB frames, four input-stack RGB images, one principal T2 U-Net result, and four T1–T4 temporal-mask inputs.

## Data provenance

The RGB and masks are from 0353-2-B3, archived V35 U-Net binary segmentation and the matching V66-pipeline RGB. Original full-field files were verified against the archived manifest. asset_verification.json records their SHA256 values and parent ROI x=1530:1830, y=590:850. The final square panels use the central 260×260-pixel crop of that parent window, equivalent to x=1550:1810, y=590:850. Display coloring turns existing foreground mask pixels red on black, without changing the binary mask. No training, registration, new segmentation or new accuracy evaluation was performed.

The original figure's U-Net Dice label 0.8788 was checked against the existing V35 value 0.8787985060232514. It is the archived evaluation Dice, not a Dice recomputed on the illustrative local crop. The bar plot and large blue/green/orange root panel remain qualitative illustrations; they are not new quantitative data or new root-instance validation results. The workflow diagram is a high-level illustration, not a replacement for the implementation and evaluation protocol.

## Files

- Root_workflow_real_images_TimesNewRoman_white.jpg / .png / .pdf / .tiff: final publication composition.
- T1–T4_actual_RGB.png and T1–T4_actual_UNet_red_mask.png: exact scientific raster assets.
- publication_composition.json: every placed panel, frame and pixel-array checksum.
- publication_verification.json: source, font, color, boundary and image checks.
- asset_verification.json: original-file hashes and verified Dice.
- workflow_illustrative_base.png: the edited schematic asset used behind the real-data panels.
- compose_publication_figure.py / verify_publication_figure.py: full composition and verification code.

Canvas width is 240 mm with the source workflow's aspect ratio. Raster files are exported at 600 dpi. PDF text is vector Times New Roman regular/bold/italic, 9–14 pt, all black. Data images and schematic drawings are raster assets; the PDF is not claimed to be an entirely vector illustration. The untouched original generated image also remains in /workspace/generated_images in the current environment.

The archive omits licensed font binaries. To reproduce, provide Times New Roman.ttf, Times New Roman Bold.ttf and Times New Roman Italic.ttf locally:

```bash
python compose_publication_figure.py --fonts-dir /path/to/fonts
python verify_publication_figure.py
```

The archive includes the exact local crops used for assembly. Rechecking full-field hashes additionally requires the full-field source paths listed in asset_verification.json; the PDF and local-panel checks do not depend on those files being present.
