# All active minirhizotron series: original context and common fields

The combined PDF has **18 pages**, one page for each active image series, grouped into **8 tubes** through PDF bookmarks. It contains **66 complete original RGB views and 66 corresponding published common-field RGB views**. The four fixed columns are T1–T4. Every page has its own English figure caption; editable Chinese and English captions are also supplied in captions_CN_EN.txt.

The template follows the supplied example:

- A: complete original RGB context, with the existing observation polygons in cyan and the selected local window in gold.
- B: the complete archived common observation field, with the same gold local window in the T1 reference coordinates.

Gold windows use the existing manuscript figure specifications, mapped to the original images using the saved transformations. Multiple observation bands retain their archived order and white non-data separator rows. The pages keep each image's aspect ratio and full field of view. Their width is 190.5 mm, with height adjusted to the image aspect ratio and caption length. Text is black Times New Roman, 8–11 pt. Text and boxes remain vector objects; RGB display copies are embedded at up to 600 dpi, JPEG quality 97. Original files are unchanged. No new registration, model inference, image enhancement or training was performed.

## Completeness and scientific scope

The current active manifest defines 18 series across 8 tubes, with 12 complete four-date series and 6 three-date series. All 66 available original frames are shown. Six unavailable T1–T4 slots have explicit Not acquired placeholders in both rows, never replacement images.

The caption and observed_date_pair_index.csv enumerate all 48 links between successive available observations: 43 are consecutive numbered periods, and 5 bridge a missing acquisition. Gap links are explicitly labeled rather than presented as T1–T2 or another nonexistent adjacent pair. The four-date layout is a display of the existing frozen localization; it is not a new matching experiment.

Canonical Q1-1-Z2 replaces its legacy source identifier 0353-1-Z2 and belongs to tube Q1-1. Its corrected root-band RGBs are used; the T2–T3 link remains coarse and unverified and is disclosed in that page's caption. Review-required observation bands retain dashed cyan boundaries. Corrected names and high display resolution do not validate unresolved geometry. The PDF does not claim manual-landmark registration accuracy or root-instance identity validation.

Q1-3-Z3 is excluded according to the current project identity policy and is not included as a current figure. Historical files remain untouched. The frozen identity policy, active manifests and saved geometry are bundled under provenance/.

## Files

- All_series_T1_T4_RGB_atlas.pdf: complete 18-page illustrated PDF, supplied as a separate download.
- captions_CN_EN.txt / captions.json: editable captions for each series and PDF page.
- observed_date_pair_index.csv: all 48 observed-date links and their page numbers.
- page_source_manifest.json: original/common source hashes, projected box coordinates and page dimensions.
- originals_verification.json / RGB_verification.json: checksums of all 132 source files used.
- verification.json: completeness, source hash, font/color, bounding-box and page checks.
- All_series_contact_sheet.jpg: quick overview of all 18 pages.
- build_atlas.py / fetch_sources.py / pdf_text_mapping.py / verify_atlas.py: complete retrieval, composition and validation scripts.

The small code-and-captions ZIP omits the large PDF and raw RGB collection. The scripts retrieve the exact published source objects by SHA256. No font binaries or credentials are included.

## Reproduce

Use the versions in requirements.txt. Obtain a licensed copy of Times New Roman regular and bold locally, with filenames Times New Roman.ttf and Times New Roman Bold.ttf in one directory.

```bash
python fetch_sources.py
python build_atlas.py --fonts-dir /path/to/fonts
python verify_atlas.py
```

fetch_sources.py first reuses checksum-matching local originals when available. Missing originals are selectively extracted from the checksum-pinned public V-.zip and corrected Q1.zip source releases. Only active source frames are extracted. Existing registered RGB is downloaded through the standard public GitHub LFS interface, with SHA256 and byte-count checks. It does not recreate or estimate registration geometry.

The source releases are https://github.com/Annalee811/LI/releases/tag/v-change-data. Registered RGB object IDs and the current dataset policy are pinned by the bundled active_images.csv and identity_policy.json. The geometry snapshot and its source-specific evidence are in provenance/geometry_registry_current.json. The two copied historical figure scripts preserve the original gold-window specification for audit; only build_atlas.py is needed to make this two-row atlas.
