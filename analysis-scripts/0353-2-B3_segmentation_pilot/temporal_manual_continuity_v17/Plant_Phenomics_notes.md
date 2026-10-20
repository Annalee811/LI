# Plant Phenomics reporting notes — V14-based manual continuity

Report the manual temporal classes and U-Net morphology separately. This run does not replace temporal classes using U-Net predictions. The exact original manual project, versioned morphology reference, unknown-pixel coverage and original audit are retained. The root morphology is model-adjusted and is not an independent manual test truth.

All three temporal heads were trained32epochs, but validation selected the new U-Net head and retained V14 emissions for DeepLab/SegFormer. Describe their final learned routing/decoder accurately. Binary root networks remain frozen. The paired temporal heads share a U-shaped design; do not describe this as end-to-end temporal fine-tuning of all three backbone architectures.

Use the matched before/after table for version comparisons. The user's V14 teacher-reference Dice values are a separate historical result, not the same reference/inference protocol. Include original-human audit and class-specific errors, especially the rare growth class. Root-instance IDs are not independently labeled: zero NEW/GROWTH mixing on accepted axes is enforced and cannot be reported as100% root-identity accuracy.

All dates at a window share a split;18test date-pairs are correlated views of6locations in one field. Historical full-field exposure and context across window borders preclude a fresh blind independent claim. External test images:0. Skeleton pixel counts are uncalibrated mask-derived lengths, not independent physical manual measurements.

Recommended main figures: integrated four-date RGB/reference/model atlas, matched metrics, crossing before/after. Supplements:S1curves,S2transition Dice,S3length agreement,S4test-window differences,S5errors,S6confusions,S7crossings,S8axis constraint counts. Green errors are correct class,pink false positives,orange false negatives. Disclose the reference version and held-out window selection in the Methods.

Use the immutable GitHub commit for code/data availability. Deposit an archival release with a DOI if available; do not invent a DOI. Check current journal upload requirements before submission. No Word file is required.

Real methodological references, if discussed in the manuscript:

- Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015. https://doi.org/10.1007/978-3-319-24574-4_28
- Chen LC et al. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV 2018. https://doi.org/10.1007/978-3-030-01234-2_49
- Xie E et al. SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021. https://arxiv.org/abs/2105.15203
- Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network. 2015. https://arxiv.org/abs/1503.02531
- Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. Lancet 1986. https://doi.org/10.1016/S0140-6736(86)90837-8

