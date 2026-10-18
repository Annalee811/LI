# B3 三模型根段连续性优化 v13

在v12三个时序头上继续训练，加入同类人工根段的骨架连续性监督和预测骨架链解码，抑制沿根反复换色。保留持续段与末端延伸的真实界限，新生侧根单独分类。交叉点不合并身份；死亡不纳入。按原图视觉复核修订4处328个时序像素，原始人工文件保留；前后模型均按同一修订标签评估。

|模型|持续Dice 旧→新|新生Dice 旧→新|生长Dice 旧→新|宏Dice 旧→新|
|---|---:|---:|---:|---:|
|U-Net|0.7308 → 0.7308|0.6151 → 0.6151|0.1357 → 0.1357|0.4939 → 0.4939|
|DeepLabV3+|0.6404 → 0.6404|0.5307 → 0.5307|0.0000 → 0.0000|0.3904 → 0.3904|
|SegFormer-B0|0.6902 → 0.6902|0.5740 → 0.5740|0.0614 → 0.0614|0.4419 → 0.4419|

三个模型均完成训练试验。按C3的类别Dice和沿骨架换色次数选择网络权重与连续性解码强度，具体采用轮次见chain_selection.json；允许保留旧网络。完整四期图是全标注最终头的拟合展示，不是独立预测准确率。根系对象身份和交叉追踪仍未验证。C4是已反复检查的同位置区域，所有变化包括退步均保留。

|ZIP下载|大小MB|
|---|---:|
|[0353-2-B3_temporal_v13_annotation_review.zip](0353-2-B3_temporal_v13_annotation_review.zip)|34.1|
|[0353-2-B3_temporal_v13_paper_figures.zip](0353-2-B3_temporal_v13_paper_figures.zip)|80.1|
|[0353-2-B3_temporal_v13_results_unet.zip](0353-2-B3_temporal_v13_results_unet.zip)|58.4|
|[0353-2-B3_temporal_v13_results_deeplab.zip](0353-2-B3_temporal_v13_results_deeplab.zip)|58.3|
|[0353-2-B3_temporal_v13_results_segformer.zip](0353-2-B3_temporal_v13_results_segformer.zip)|57.3|
|[0353-2-B3_temporal_v13_training_data.zip](0353-2-B3_temporal_v13_training_data.zip)|90.8|
|[0353-2-B3_temporal_v13_weights_and_scripts.zip](0353-2-B3_temporal_v13_weights_and_scripts.zip)|75.2|

[方法与复现](README_method_and_reproduce.md)。训练数据包包含原始人工JSON与可导入网页的修订JSON、四期上半部、原始标签及有记录的时序修订标签、实际连接区训练裁剪及剔背景根RGB；增强与完整抽样记录保留。权重脚本包包含三模型新旧时序权重和完整源码、U-Net/SegFormer根网络权重。冻结第一阶段输入和原始DeepLab根权重复用已校验的v10包：

- [0353-2-B3_temporal_v10_frozen_features_unet.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_features_unet.zip)
- [0353-2-B3_temporal_v10_frozen_features_deeplab.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_features_deeplab.zip)
- [0353-2-B3_temporal_v10_frozen_features_segformer.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_features_segformer.zip)
- [0353-2-B3_temporal_v10_frozen_DeepLab_weights.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_DeepLab_weights.zip)

预测始终使用前期模型根掩膜，不使用人工前期掩膜。每个ZIP低于GitHub文件限制，SHA256见download_manifest.json。

本次运行记录：三模型各完成24轮修订标注训练试验；新增候选未进一步提高开发验证表现，评估头保留v12权重。三个final_head另以修订标签完成全部上半部24轮拟合。所有验证指标仅对应evaluation_head；拟合图不代表独立测试准确率。[本次记录](revision_training_summary.json)。
