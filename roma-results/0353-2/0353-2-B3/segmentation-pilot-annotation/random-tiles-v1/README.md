# 0353-2-B3：二维小图随机划分实验

只取四期已人工标注的上半部分；直接切256×256原像素小图，再固定随机分配。没有C0–C4大区分组。
36个图块位置：25训练、5验证、6内部测试；四期对应100、20、24张RGB小图。同一坐标的四期始终同组，零跨集合像素重叠。
三模型从零初始化，无旧B3或ImageNet权重。人工标签未修改，不按根系多少/难易筛图。验证集选权重和阈值，测试集仅评估。
这是同一位置的内部预实验；外部独立位置/根管测试图像为0。新的随机小图测试不能与旧C3/C4数值混合平均。

|模型|Precision|Recall|Dice|IoU|
|---|---:|---:|---:|---:|
|U-Net|0.6582|0.7787|0.7134|0.5545|
|DeepLabV3+|0.5544|0.6731|0.6080|0.4368|
|SegFormer-B0|0.5901|0.6277|0.6083|0.4371|

## 下载

- [0353-2-B3_random_tiles_v1_paper_figures.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_paper_figures.zip) (83.5 MiB)
- [0353-2-B3_random_tiles_v1_four_period_RGB_comparison.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_four_period_RGB_comparison.zip) (66.1 MiB)
- [0353-2-B3_random_tiles_v1_background_removed_full_and_local.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_background_removed_full_and_local.zip) (32.0 MiB)
- [0353-2-B3_random_tiles_v1_PlantPhenomics_guide.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_PlantPhenomics_guide.zip) (0.0 MiB)
- [0353-2-B3_random_tiles_v1_training_tiles_labels_RGB.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_training_tiles_labels_RGB.zip) (38.9 MiB)
- [0353-2-B3_random_tiles_v1_evaluation_unet.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_evaluation_unet.zip) (92.6 MiB)
- [0353-2-B3_random_tiles_v1_evaluation_deeplab.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_evaluation_deeplab.zip) (93.6 MiB)
- [0353-2-B3_random_tiles_v1_evaluation_segformer.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_evaluation_segformer.zip) (93.2 MiB)
- [0353-2-B3_random_tiles_v1_models_torch_scripts.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_models_torch_scripts.zip) (57.5 MiB)
- [0353-2-B3_random_tiles_v1_model_deeplab_scripts.zip](https://raw.githubusercontent.com/Annalee811/LI/d0f862968abc613dc761f12ffc4ae0cd70de8c49/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1/0353-2-B3_random_tiles_v1_model_deeplab_scripts.zip) (64.9 MiB)

论文主文建议Fig1–Fig4：随机划分、三模型测试指标、固定小图四期分割、三模型根长一致性；其他图与逐块明细作补充。英文方法、结果、图注及中文投稿说明见guide包。
根长参考由人工掩膜骨架推导，单位为像素，并非独立人工尺量。原图整幅上半部分的分割包含训练像素，只作定性展示。
所有ZIP已做CRC完整性验证，SHA256见SHA256.json；公开直链的逐字节校验结果另存于public_download_verification.json。

固定数据提交：`d0f862968abc613dc761f12ffc4ae0cd70de8c49`。已公开下载全部10个ZIP，核对完整字节数及SHA256均通过。所有11组论文TIFF为600dpi，本轮Word文件为0。
