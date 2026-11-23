SCI figure package — 2026-11-23

Six-panel figure, final size 190 x 145 mm; white background, black Times New Roman
text; 8-pt axis labels, 12-pt panel letters. JPG, PNG and lossless LZW TIFF are 600 dpi.
PDF embeds Times New Roman; SVG retains editable text and requires the font locally.
Use the vector PDF or TIFF for submission, subject to the target journal's size rules.

Source results are copied, hashed and included. No new training, inference,
registration or modification of old results was performed for this visualization.
The current figure reports the newly added conditional control experiments. Its
"Full" configuration is not a relabelling of historical V90 headline scores.

TOP ROW: internal partition-isolated module study.
BOTTOM ROW: separate paired whole-field structural-decoder study.
Do not splice the two contexts into a single before/after comparison.
Errors are sample SD over three optimization seeds, not confidence intervals or
biological replication. No statistical significance or P-values are implied.
Frozen V66 is one historical reference; current-frame-only is a collapsed negative
control under this budget. Identity-gate improvement is small and inconsistent.
Final root-identity recall remains low; cross-tube accuracy is unavailable without
human temporal ground truth. Empty annotation templates are not truth.

Results_and_figure_caption.docx contains editable English/Chinese results prose
and an English figure caption. Chinese_analysis.txt and English_results_and_caption.txt
provide plain-text copies; no tables are needed.

Rebuild:
python plot_sci_results.py --results /path/to/archived/reports   --fonts /path/to/licensed/Times-New-Roman/fonts --output /new/output
The plot uses only existing CSV/JSON files. Source copies have a source_ prefix;
to rebuild from this bundle alone, copy them to another results directory after
removing the source_ filename prefix. Fonts are not redistributed.
Python dependencies: numpy, matplotlib, Pillow; python-docx is needed only for
write_analysis.py. Sources and per-panel values are in this archive.
