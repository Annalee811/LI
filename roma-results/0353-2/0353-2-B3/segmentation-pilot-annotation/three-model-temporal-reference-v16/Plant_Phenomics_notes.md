# Plant Phenomics — reporting and data availability notes

Use the integrated TIFF for the four-date qualitative result, class-Dice figure for temporal error distribution, and supplementary S1–S6 for optimization histories, transition Dice, skeleton-length agreement, test-window variability, local errors and confusion matrices. Use the original-human audit alongside teacher-reference agreement. Full-field figures include fitting pixels and cannot establish external generalization.

Describe the revised reference as U-Net-derived morphology and temporal classes, reviewed only if actual review occurs. It is not independent manual ground truth. The U-Net used to generate this reference is also the starting checkpoint, so perfect teacher-reference agreement is circular. Do not describe a score of 1 as evidence of independently perfect segmentation. Do not claim external testing: the only labeled position is 0353-2-B3, and the earlier root models/warm starts have prior exposure to this field. All four dates at one spatial window share a split. The split provides an internal development comparison rather than a cross-tube evaluation.

Suggested methods wording: We used previously trained binary root detectors to suppress background and fine-tuned corresponding paired temporal classification heads. A versioned reference was constructed from the latest U-Net whole-axis predictions; the original manual annotations were preserved for a separate audit. Three temporal heads were fine-tuned with identical paired spatial windows and augmentations for 24 epochs, and checkpoints were selected using internal validation macro Dice. Full-image logits were blended before applying a root-axis decoder that constrained newly observed axes and persistent axes with terminal extension. The four-date baseline and subsequent persistent, newly observed and extension classes were evaluated separately. The pretrained binary detectors were frozen. Temporal heads used PyTorch; the DeepLabV3+ binary model used the supplied TensorFlow implementation. All results are from one position; independent cross-tube accuracy remains unassessed.

Do not report skeleton pixel counts as physical root lengths without calibration, or as independent manual length measurements. Report identity R², OLS R² and RMSE separately; correlation alone is not agreement. Do not report confidence intervals that treat correlated dates/windows as independent plants. Root IDs at crossings are putative geometry and require review.

For a submission, include the real GitHub archive commit in Code Availability, the scripts/weights/data archive links in Data Availability, a complete reference-version description in Methods, and the supplemental figure archive as supplementary material. A DOI-bearing archival release (e.g. Zenodo) is preferable to a mutable branch for long-term availability. Only state a DOI once it actually exists. Review the current journal author instructions for upload-specific limits; no unverified journal-specific requirements are asserted here.

Real methodological references, if discussed in the manuscript:

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://doi.org/10.1007/978-3-319-24574-4_28
- Chen LC et al. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV 2018. https://doi.org/10.1007/978-3-030-01234-2_49
- Xie E et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021. https://arxiv.org/abs/2105.15203
- Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network. 2015. https://arxiv.org/abs/1503.02531
- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet 1986. https://doi.org/10.1016/S0140-6736(86)90837-8

These references describe architectures, teacher-based learning and measurement agreement; they do not validate the present model-generated reference as manual truth.
