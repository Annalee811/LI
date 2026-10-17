# B3 三模型根系时序分割 v10

在各自v9根系/背景分割基础上，先剔除背景，再训练三个独立时序分类头；根背景主干冻结。T1基础根；T2/T3/T4分别相对前一期，分类为持续、新生、生长。死亡不纳入。

人工标注未更改，仅上半部。原始4张图、3个相邻转换；训练216对重叠切块、验证72对、同位置重复基准72对，外部独立测试图0。完整展示采用另训的全上半部最终头，是拟合展示；C3/C4评估来自仅在C0–C2训练的评估头。

|模型|C3持续Dice|C3新生Dice|C3生长Dice|C3宏Dice|C4宏Dice|
|---|---:|---:|---:|---:|---:|
|U-Net|0.7259|0.5935|0.0929|0.4707|0.3858|
|DeepLabV3+|0.6343|0.5068|0.0000|0.3804|0.2707|
|SegFormer-B0|0.6890|0.5652|0.0602|0.4381|0.2814|

验证选优：**U-Net**。选优只用C3；C3参与轮数和校准选择，C4为已反复查看的同位置基准，两者均不能代替跨根管独立测试。各类支持量、Precision/Recall/Dice/IoU、混淆计数、校准前后数据均保存。生长段指标不足时如实报告，不等同于已验证的高精度对象追踪。

|下载ZIP|大小MB|
|---|---:|
|[0353-2-B3_temporal_v10_frozen_features_unet.zip](0353-2-B3_temporal_v10_frozen_features_unet.zip)|74.6|
|[0353-2-B3_temporal_v10_frozen_features_deeplab.zip](0353-2-B3_temporal_v10_frozen_features_deeplab.zip)|78.6|
|[0353-2-B3_temporal_v10_frozen_features_segformer.zip](0353-2-B3_temporal_v10_frozen_features_segformer.zip)|79.3|
|[0353-2-B3_temporal_v10_paper_figures.zip](0353-2-B3_temporal_v10_paper_figures.zip)|35.9|
|[0353-2-B3_temporal_v10_training_data.zip](0353-2-B3_temporal_v10_training_data.zip)|96.5|
|[0353-2-B3_temporal_v10_results_unet.zip](0353-2-B3_temporal_v10_results_unet.zip)|56.5|
|[0353-2-B3_temporal_v10_results_deeplab.zip](0353-2-B3_temporal_v10_results_deeplab.zip)|57.0|
|[0353-2-B3_temporal_v10_results_segformer.zip](0353-2-B3_temporal_v10_results_segformer.zip)|57.5|
|[0353-2-B3_temporal_v10_weights_and_scripts.zip](0353-2-B3_temporal_v10_weights_and_scripts.zip)|63.2|
|[0353-2-B3_temporal_v10_frozen_DeepLab_weights.zip](0353-2-B3_temporal_v10_frozen_DeepLab_weights.zip)|74.2|

方法、复现和解释见[README_method_and_reproduce.md](README_method_and_reproduce.md)。三个frozen_features包合并解压为ROOT_V9，数据包解压为RUN/data；它们是复现所需的冻结第一阶段概率与根掩膜。权重脚本包包含全部三个时序头、U-Net/SegFormer根网络和完整源码，原始DeepLab根权重另包提供。全部ZIP小于GitHub单文件限制，下载后可按download_manifest.json核对SHA256。
