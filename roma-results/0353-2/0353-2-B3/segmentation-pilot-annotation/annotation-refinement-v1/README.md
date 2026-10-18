# 0353-2-B3 修订参考与本轮最终三模型结果

根系/背景任务，仅四期已标注上半部分。原始人工标注完整保留；修订版参考旧U-Net与RGB管状结构支持，新增根宽和漏标记录可逐像素追溯。它不是独立人工真值。
200训练、20验证、24内部测试切块；25个训练基础位置各增加一个12像素平移窗口。四期同组，新增窗口排除验证/测试像素。训练窗之间有重叠，不是200张独立原始图。模型与阈值按原始验证标签选择；本交付图表统一对修订参考计算一致性。原始标签与原始保留测试指标另存供核查，不混入旧模型图。修订参考并非独立人工真值。
U-Net、原上传TensorFlow DeepLabV3+及SegFormer均从对应旧权重微调20轮；若原始验证不改善，可以保留第0轮原权重。未修改原始RGB颜色或尺寸。

|模型|Precision|Recall|Dice|IoU|
|---|---:|---:|---:|---:|
|U-Net|0.9448|0.8316|0.8846|0.7931|
|DeepLabV3+|0.7847|0.6882|0.7333|0.5789|
|SegFormer-B0|0.8132|0.6549|0.7255|0.5693|

## ZIP下载

- [0353-2-B3_final_paper_figures.zip](0353-2-B3_final_paper_figures.zip) (83.6 MiB)
- [0353-2-B3_final_four_period_RGB_comparison.zip](0353-2-B3_final_four_period_RGB_comparison.zip) (39.4 MiB)
- [0353-2-B3_final_full_upper_RGB_comparison.zip](0353-2-B3_final_full_upper_RGB_comparison.zip) (90.3 MiB)
- [0353-2-B3_final_background_removed_full_local.zip](0353-2-B3_final_background_removed_full_local.zip) (28.7 MiB)
- [0353-2-B3_final_PlantPhenomics_guide.zip](0353-2-B3_final_PlantPhenomics_guide.zip) (0.0 MiB)
- [0353-2-B3_final_training_tiles_annotations_RGB.zip](0353-2-B3_final_training_tiles_annotations_RGB.zip) (50.6 MiB)
- [0353-2-B3_final_evaluation_unet.zip](0353-2-B3_final_evaluation_unet.zip) (74.1 MiB)
- [0353-2-B3_final_evaluation_deeplab.zip](0353-2-B3_final_evaluation_deeplab.zip) (75.7 MiB)
- [0353-2-B3_final_evaluation_segformer.zip](0353-2-B3_final_evaluation_segformer.zip) (75.3 MiB)
- [0353-2-B3_final_models_torch_scripts.zip](0353-2-B3_final_models_torch_scripts.zip) (57.5 MiB)
- [0353-2-B3_final_model_deeplab_scripts.zip](0353-2-B3_final_model_deeplab_scripts.zip) (65.1 MiB)

表中是相对修订参考的一致性，不是独立人工准确率。主图Fig2三模型指标、Fig3四期带背景对比、Fig4三模型根长回归；S1训练曲线、S2逐期Dice、S3根长偏差、S4图块差异、S5误差定位。S5绿色TP、粉色FP、橙色FN均相对修订参考。根长由修订掩膜骨架推导，单位像素，不是独立人工尺量。外部独立测试图像数为0。
ZIP包含CRC和SHA256校验，完整脚本、可编辑矢量统计图及600dpiTIFF。没有Word文件。公开下载的逐字节验证见public_download_verification.json。

## 固定版本直接下载

- [0353-2-B3_final_paper_figures.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_paper_figures.zip)
- [0353-2-B3_final_four_period_RGB_comparison.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_four_period_RGB_comparison.zip)
- [0353-2-B3_final_full_upper_RGB_comparison.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_full_upper_RGB_comparison.zip)
- [0353-2-B3_final_background_removed_full_local.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_background_removed_full_local.zip)
- [0353-2-B3_final_PlantPhenomics_guide.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_PlantPhenomics_guide.zip)
- [0353-2-B3_final_training_tiles_annotations_RGB.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_training_tiles_annotations_RGB.zip)
- [0353-2-B3_final_evaluation_unet.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_evaluation_unet.zip)
- [0353-2-B3_final_evaluation_deeplab.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_evaluation_deeplab.zip)
- [0353-2-B3_final_evaluation_segformer.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_evaluation_segformer.zip)
- [0353-2-B3_final_models_torch_scripts.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_models_torch_scripts.zip)
- [0353-2-B3_final_model_deeplab_scripts.zip](https://raw.githubusercontent.com/Annalee811/LI/2f3fa0bdca49e37833cb1a83b3b0a75464b59dcf/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/annotation-refinement-v1/0353-2-B3_final_model_deeplab_scripts.zip)
