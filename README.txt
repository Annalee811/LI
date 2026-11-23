V90 complete workflow schematic
Created: 2026-11-23

Landscape, 190 x 112 mm. White background and black Times New Roman text.
JPG, PNG and LZW TIFF are exported at 600 dpi.
PDF contains embedded Times New Roman fonts and vector shapes/arrows/text.
SVG retains editable text and vector elements; install Times New Roman for consistent rendering.

Four aligned stages:
A: registered common-field RGB inputs and shared frozen preprocessing;
B: the learned network, with ConvLSTM + GAT grouped and Gate inside GAT;
C: sequential Axis and Junction decoding guided by fixed neural evidence;
D: final New / Existing mapping, RGB overlay, axis IDs and confidence flags.

Reproduce using Python with numpy and matplotlib:
python draw_workflow.py --fonts /path/to/fonts
The directory must contain Times New Roman.ttf, Times New Roman Bold.ttf and Times New Roman Italic.ttf.
Font files are not redistributed.

Workflow_RGB_and_mask_cache.npz preserves four original B3 RGB crops and the original decoded three-class T2 mask. The drawing script performs the New / Existing grouping and scientific overlay during figure rendering. Sources, crop coordinates, selection rule and hashes are recorded in Image_provenance.json. No original file was overwritten or moved.

Figure_caption_EN.txt supplies the English caption and interpretation notes.
图注与流程说明.txt supplies the Chinese caption and suggested paper title.
Diagram_checks.json and Verification.json document layout, resolution, fonts and source preservation.

The diagram is a method schematic, not an accuracy result. P = persistent, N = new, E = extending. Existing = P + E. Internal network outputs and structural decoder remain three-class in the current implementation. Predictions use offline sequence evidence, potentially including later frames. Root masks stay geometrically fixed during decoding.
