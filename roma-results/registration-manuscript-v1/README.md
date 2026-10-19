# Registration manuscript source code

[Download the original registration and manuscript-figure scripts](registration_manuscript_original_scripts_v1.zip) (0.29 MiB; 98 files).

[Current reproducible entry point](../../analysis-scripts/registration_manuscript_v1/README.md). Includes RoMa source/license, RootSIFT/RANSAC/region-localization source, canonical frozen geometry, model SHA256 pins, dependencies, four single-page TIFF scripts, and published table snapshot. No images or model weights are duplicated.

The active data are 18 positions and 66 original frames. Q1-3-Z3 is excluded; 0353-1-Z2 is corrected to Q1-1-Z2. Source-specific historical estimation paths remain documented for audit. The current wrapper rebuilds the selected semi-automatic geometry, not a new fully automatic localization. Q1-1-Z2 remains provisional.

Checks: 66 original SHA256 values, all 66 rebuilt registered RGB SHA256 values, all four regenerated TIFF byte hashes, isolated table-summary equality, and analytic dense-coordinate transport/seam tests passed. ZIP CRC and full archive SHA256 are verified. This code update did not retrain models or rerun the 48 dense inference pairs.
