# V90 full-field temporal labels on original RGB — 0353-2-B3

These are the archived final V90 temporally optimized learned/structural/reviewed labels over their matching T1–T4 RGB backgrounds. The complete available registered field is 2541×1108 pixels per date. No local crop, extra registration, retraining, new inference, smoothing, root enlargement or label reassignment is performed. This is the complete archived model field, rather than a claim that the images reproduce the original unregistered camera sensor extent.

Colors follow the latest workflow figure: newly emerged = yellow; elongating = green; persistent = blue. T1 is a baseline and its existing root class is shown in white; its roots are not classified as newly emerged, elongating or persistent relative to an unavailable prior date. Background (code 0) is copied exactly from RGB. Existing uncertain/ignored pixels (code 255), if present, would be gray; none are present in these four released maps. No new uncertainty classification is inferred.

The PNG folder contains the full native-resolution, lossless RGB overlays without captions or added borders. Every background pixel equals the corresponding original RGB value. JPG contains high-quality preview copies, which are lossy. The four-period overview uses black actual Times New Roman, a white surrounding canvas and a color legend; images retain their original backgrounds. The source RGB and original class-label PNGs are included unchanged in sources. Source hashes match the archived manifests at commit 4615bcbd37442a3b11e5ad4533242df95f237fb6. Export details, class counts, colors and pixel-array hashes are in export_manifest.json.

V90 is the final learned/structured workflow with temporal/identity constraints and a released manual-review correction. These outputs are not pure neural-only predictions or a new independent validation. Class colors encode temporal states, not unique root-instance identity at crossings. This export preserves the original class maps and does not establish a new crossing-ownership validation.

To reproduce, install the versions in requirements.txt and provide licensed Times New Roman.ttf and Times New Roman Bold.ttf locally (font binaries are not redistributed):

```bash
python export_v90_overlays.py --fonts-dir /path/to/fonts
```

The script validates archived file hashes, native dimensions and labels, exports exact lossless overlays, and checks background equality and PDF fonts. Historical source images, predictions and prior results remain unchanged.
