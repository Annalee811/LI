# 时序基线、模块消融与跨微根管测试材料

已实际完成5种配置×3个随机种子×100次整图优化，共15组实验、1500次更新。实验冻结上游分割与V66先验，在B3内部重新训练时序/图模块；权重依据验证集选择。原项目和旧权重保持原样。

完整配置内部宏平均Dice为63.71±0.35%，冻结V66参照为62.56%。去掉ConvLSTM、GAT和学习式身份门控后分别为63.42%、62.95%和63.69%。当前帧负对照在本轮预算下退化为持续类预测，不能作为充分调优的强单帧基线。

另一个配对结构解码实验在同一完整视野神经输出上比较：原始类别头63.70±0.34%，加入根轴约束73.51±0.50%，加入根轴与连接处约束76.71±1.99%。该实验与分区隔离的模块主实验具有不同推理上下文，请分别使用。

跨管已固定0353-3、A017-3、Q1-2三个微根管，共7个位置26幅图像，并完成预测。其他微根管尚无人工时序真值，独立Dice、IoU保持未计算。空模板全255，只表示未标注。预测入口保留已记录的特征适配近似，跨管性能需人工真值检验。

随机种子SD表示优化波动，不是生物学重复。完整表格保留全部指标，未生成显著性P值。

## 下载

所有链接固定到同一个归档提交，文件均可独立下载。

- [完整实验：代码、固定特征、15组最优与续训权重、日志、指标及高清图](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/V90_actual_experiments_code_weights_results.zip)（55.67 MB）
- [全部指标与独立样本清单（Excel）](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Baseline_ablation_and_test_index.xlsx)（0.02 MB）
- [内部基线与模块消融图（高清JPG）](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Baseline_ablation_SCI.jpg)（1.79 MB）
- [根轴与连接处结构解码对照图（高清JPG）](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Structured_decoder_SCI.jpg)（0.93 MB）
- [7个位置、四期跨管原图与标记预览（7页PDF）](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Independent_prediction_preview.pdf)（7.16 MB）
- [跨管预测类别图：390张新模型结果及52张冻结参照](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Independent_predictions_442_masks.zip)（28.09 MB）
- [完整中文实验报告](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Experiment_report_CN.txt)（0.01 MB）
- [独立测试RGB与空标注模板：0353-3](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Annotation_RGB_templates_0353-3.zip)（35.22 MB）
- [独立测试RGB与空标注模板：A017-3](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Annotation_RGB_templates_A017-3.zip)（53.15 MB）
- [独立测试RGB与空标注模板：Q1-2](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Annotation_RGB_templates_Q1-2.zip)（77.34 MB）
- [内部基线与模块消融图（矢量PDF）](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Baseline_ablation_SCI.pdf)（0.07 MB）
- [复现结构解码的原始B3输入与拓扑](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Decoder_inputs.zip)（51.95 MB）
- [重放跨管预测的固定输入特征：0353-3](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Independent_inputs_0353-3.zip)（9.52 MB）
- [重放跨管预测的固定输入特征：A017-3](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Independent_inputs_A017-3.zip)（16.96 MB）
- [重放跨管预测的固定输入特征：Q1-2-B3](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Independent_inputs_Q1-2-B3.zip)（43.26 MB）
- [重放跨管预测的固定输入特征：Q1-2-Z2](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Independent_inputs_Q1-2-Z2.zip)（21.28 MB）
- [重放跨管预测的固定输入特征：Q1-2-Z3](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Independent_inputs_Q1-2-Z3.zip)（50.13 MB）
- [结构解码对照图（矢量PDF）](https://raw.githubusercontent.com/Annalee811/LI/4a4e61a9e14ece88f6c8ac5bb3bc044a9170a01e/Structured_decoder_SCI.pdf)（0.06 MB）

## 复现与评价

先解压完整实验包，按README.txt安装Python 3.12及依赖。`code/reproduce_study.py --output 新空目录`可重跑内部控制训练；最优权重及续训权重均已提供。结构解码另下载Decoder_inputs.zip。跨管重放另下载对应的Independent_inputs包，将其中independent_inputs目录合并。

三个RGB模板包解压后，将7个位置文件夹合并到同一个annotation-root目录。完成来自RGB的人工标注后，在完整26幅清单中填写human_complete、annotator、annotation_source，使用evaluate_independent.py评价。评价入口拒绝空模板；按微根管汇总，不把时期或小图当作独立重复。

原生类别PNG编码为0背景、1持续、2新出现、3延伸、4首个观测时期根区。编码值较低，直接打开可能显暗，请使用PDF叠加预览；原生分辨率掩膜另存。缺失日期未补图。新的跨管预测没有读取人工时序真值。

跨管神经预测与整图解码的便携入口已分别重跑检查，预测及归属数组与正式输出一致。逐文件SHA256见ARTIFACT_SHA256.json，原始执行代码哈希记录在locked_protocol.json。
