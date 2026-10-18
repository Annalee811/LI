# B3 三模型连接点与时序类别优化 v11

在v10三个时序头上继续训练：新生、持续、生长全部纳入；根背景模型与人工标注不变。没有将物理连通或交叉直接作为同一类别的规则。死亡不纳入。

|模型|持续Dice 旧→新|新生Dice 旧→新|生长Dice 旧→新|宏Dice 旧→新|
|---|---:|---:|---:|---:|
|U-Net|0.7259 → 0.7280|0.5935 → 0.6010|0.0929 → 0.0946|0.4707 → 0.4745|
|DeepLabV3+|0.6343 → 0.6343|0.5068 → 0.5068|0.0000 → 0.0000|0.3804 → 0.3804|
|SegFormer-B0|0.6890 → 0.6890|0.5652 → 0.5652|0.0602 → 0.0602|0.4381 → 0.4381|

本轮最终接受U-Net的更新；DeepLab与SegFormer未同时满足六项互误保护条件，部署头保留旧版（训练试验与未采用权重仍保留）。这是同位置C3验证指标，C3参与选优。C4重复区域、连接点及前期根尖区的所有三类变化和三对混淆均在CSV和图中列出，包括退步，不作为跨根管独立测试。完整四期图是全标注最终头的拟合展示；模型仍不等于已验证的根对象身份追踪。

|ZIP下载|大小MB|
|---|---:|
|[0353-2-B3_temporal_v11_paper_figures.zip](0353-2-B3_temporal_v11_paper_figures.zip)|40.4|
|[0353-2-B3_temporal_v11_results_unet.zip](0353-2-B3_temporal_v11_results_unet.zip)|57.9|
|[0353-2-B3_temporal_v11_results_deeplab.zip](0353-2-B3_temporal_v11_results_deeplab.zip)|57.0|
|[0353-2-B3_temporal_v11_results_segformer.zip](0353-2-B3_temporal_v11_results_segformer.zip)|57.3|
|[0353-2-B3_temporal_v11_training_data.zip](0353-2-B3_temporal_v11_training_data.zip)|90.1|
|[0353-2-B3_temporal_v11_weights_and_scripts.zip](0353-2-B3_temporal_v11_weights_and_scripts.zip)|67.4|

[方法与复现](README_method_and_reproduce.md)。训练数据包包含原始人工JSON、四期上半部、未改动标签、实际连接区训练裁剪及剔背景根RGB；增强与完整抽样记录保留。权重脚本包包含三模型新旧时序权重和完整源码、U-Net/SegFormer根网络权重。冻结第一阶段输入和原始DeepLab根权重复用已校验的v10包：

- [0353-2-B3_temporal_v10_frozen_features_unet.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_features_unet.zip)
- [0353-2-B3_temporal_v10_frozen_features_deeplab.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_features_deeplab.zip)
- [0353-2-B3_temporal_v10_frozen_features_segformer.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_features_segformer.zip)
- [0353-2-B3_temporal_v10_frozen_DeepLab_weights.zip](https://raw.githubusercontent.com/Annalee811/LI/685ae232bdcc29806944947d658a3d6779313b1c/roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/three-model-temporal-v10/0353-2-B3_temporal_v10_frozen_DeepLab_weights.zip)

预测始终使用前期模型根掩膜，不使用人工前期掩膜。每个ZIP低于GitHub文件限制，SHA256见download_manifest.json。
