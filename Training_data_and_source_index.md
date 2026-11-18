# 训练资料与数据索引

本次任务没有重新训练或修改原始权重。人工标注用于监督训练与评价；新位置预测不读取人工标注。

原模型、训练脚本、原始训练资源与配置：

https://github.com/Annalee811/Temporal-segmentation-model/tree/4615bcbd37442a3b11e5ad4533242df95f237fb6

资源下载使用该仓库的 tools/materialize_assets.py；原资源清单包含各文件 SHA256。

本批次使用当前数据清单中的18个拍摄位置、66幅配准RGB，来自8个微根管。同一位置的所有时期保持在同一数据组。缺失日期没有补图。Q1-3-Z3未纳入；Q1-1-Z2保留配准质量标记。

本批次未建立新的训练/验证/测试划分，也没有把无人工真值的位置计入Dice或IoU评价。

All_66_frames_status.csv 和 All_18_sequences_status.csv 提供位置、时期、源RGB哈希、输出哈希和待核验状态。

18个位置均运行V90冻结权重的部署适配入口，其输出为候选分割；适配方法差异见 Input_adapter_audit.json。B3另提供原V90自动输出作独立文件包。
