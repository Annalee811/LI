# True V66/V90 two-root RGB comparison

This version uses only the actual V66 U-Net and reviewed V90 outputs from Annalee811/Temporal-segmentation-model at immutable commit 4615bcbd37442a3b11e5ad4533242df95f237fb6. V17/V18 data are not used.

The four panels show original RGB, V66 temporal segments, V90 temporal segments, and V90 candidate-axis ownership. One fixed T3 ROI in 0353-2-B3 is used: x=957:1197, y=312:492 (240×180 pixels). Only two decoded V90 candidates are highlighted: R1=axis271, R2=axis192. The same candidate foreground domain is used in every panel. Unselected roots and other background RGB pixels remain exactly unchanged. Single-valued V90 raster ownership prevents two displayed owners at one pixel.

R1 has448 newly-emerged pixels in both versions. R2 has332 persistent and484 new pixels in V66, versus434 persistent and382 extension pixels in V90. These are descriptive counts of the selected display domain, not independent accuracy or biological replicate statistics. The ROI was selected using crossing angle and local persistent/extension support; changed-pixel counts were not used to rank examples. Binary root geometry is unchanged.

Class colors are blue=persistent, amber=new, purple/magenta=extension. Axis colors are teal and terracotta and identify candidates only. No independent biological parent/lateral identity is established. A V66 instance-owner map is not fabricated: the fourth panel is explicitly V90 only, and both temporal class maps are compared on the same two V90 candidate masks. No existing class value is manually repainted to improve the example. No additional uncertainty mask is inferred; the released maps have no ignored/unknown class pixels in this crop.

V90 is the released learned/structural/manually reviewed workflow, not neural-only inference. Its reviewed labels are copied exactly from the archived final PNGs. To recover its axis-owner map, the released decoder and review stages were replayed in an isolated directory using archived exact neural arrays/logits. No training or ConvLSTM/GAT forward was rerun. All four resulting final maps matched archived V90 pixel-for-pixel. The replay record is decoder_replay_verification.json. The compact figure_inputs.npz, selected_V90_paths.npz and data_manifest.json preserve the display arrays, owner IDs, state records and hashes; figure_source_checks.json records original class/RGB source hashes. V90_archive_manifest.json provides complete immutable archive SHA256 provenance.

The full downloaded assets and decoder replay remain in this independent work directory and do not overwrite project files. They are not redundantly included in the compact figure bundle. replay_archived_decoder.py and prepare_figure_inputs.py document recovery using the published assets. Compact plotting is self-contained with make_figure.py and verify_figure.py; supply licensed Times New Roman.ttf and Times New Roman Bold.ttf (font binaries not redistributed):

python make_figure.py --fonts-dir /path/to/fonts
python verify_figure.py

The figure is210×76mm, white canvas, black actual Times New Roman, bold headings/labels/legend; 600-dpi JPG/PNG/LZW TIFF and vector PDF lettering with native raster photographs. Native photo detail remains limited by the240×180-pixel crop. The original_RGB and panel PNGs preserve exact image arrays, while PDF labels/arrows and scale are separate annotations. See the bilingual caption for scientific scope and limitations.
