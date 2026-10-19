# Q1-3-Z3 root-guided recheck

Run `PYTHON=/path/to/python bash reproduce.sh /path/to/originals /path/to/work`.
Use the supplied three original images, which are checked by SHA256. CUDA is detected automatically; this run used CPU. Requirements record the working package versions. The archive includes the exact upstream RoMa Python source and license, and model SHA256 pins; weights are downloaded through RoMa's official Torch Hub sources if absent, not duplicated in this archive.

User-guided broad regions reduce the influence of changed soil outside the observation band. Full RoMa is checked on original-color crops and CLAHE grayscale crops. Frangi saliency selects elongated structures for a global, uniform T2-to-T3 similarity; it is not a verified root segmentation. The first, least restrictive prespecified saliency threshold (0.1) is used, not whichever validation result is smallest. Original RGB pixels are never colored or warped elastically.

The additional automatic descriptor check reuses previously extracted, hash-linked images' consensus anchors, not model-fit points. That check is conditional on the earlier descriptor consensus, not independent manual landmarks. `reference_evidence` retains the exact earlier points and search record for reproducibility. All numerical errors identify their own native pixel scale.

Only T2-to-T3 has a supported pose. Do not treat the three user-guided search crops as a registered three-period sequence. T1 has no eligible bidirectional anchors even after enhancement. T1-referenced cumulative confidence and three-period common area are unknown (`null`), not zero. T4 is missing.

After fixing the pose, RoMa is run on the actual aligned original-color crops for correspondence QC; these confidence maps are conditional on that saved pose. The geometric observation mask preserves changing roots; the confidence-filtered mask is a separate matching QC product, not a root mask. There is only one supported edge, so its cumulative confidence is that edge's confidence with T2 as the reference. No valid T1 cumulative path is manufactured.

The full earlier all-series archive and other locations are unchanged. This folder/ZIP supersedes only the Q1-3-Z3 recheck entry.
