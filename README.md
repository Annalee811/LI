# 三张关键图：新生根与既有根

已将现有15组时序基线/模块实验按两类重新评价。Existing合并原持续类1与延伸类3，New保持原新生类2；先合并真值及预测，再计算混淆矩阵和指标，未平均原分类评分。模型权重与原三类验证集检查点选择保持不变，本次没有按两类重新训练网络。

图1比较当前帧负对照、冻结V66时序先验、无GAT的ConvLSTM和Full条件模块。Full两类宏Dice为89.21±0.11%，与冻结V66的89.21%接近。当前帧分类器退化为既有类预测，不是充分调优的强单帧基线。内部固定前景、有效根像素评价，不含背景。

图2展示模块配置、种子配对的消融差值，以及另一个完整视野推理上下文的结构解码干预。ConvLSTM/GAT/门控的平均差值分别为−0.056/+0.105/+0.065个百分点，不表示所有模块都有稳定增益。配对结构解码的两类宏Dice为89.12%→92.00%→92.41%。不能将两种上下文混作一个连续优化过程。

图3展示固定独立候选队列的覆盖范围，以及同视野四期RGB与两类预测。3管7位置26幅图像中有19个后续观测，尚缺人工时序真值，独立准确率未计算。主例使用重训练Full原始神经分类头、固定首个种子20261123，不包含外部完整根轴/连接处解码。冻结原V90部署适配预测另在完整7位置图集中列出，不与重训练头混称同一推理协议。

所有图稿为黑色Times New Roman、白色背景、600dpi，提供JPG/TIFF/PDF/SVG。标准差来自3个优化种子，不是生物学重复；两类宏分数不能与三类宏分数直接比较。类别评分也不能替代根实例归属验证。

## 下载

- [图1 时序基线：两类内部重新评价 JPG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_1_Temporal_baselines_two_classes.jpg)（1.38 MB）
- [图2 模块消融：两类评分与配对结构解码 JPG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_2_Module_ablations_two_classes.jpg)（1.29 MB）
- [图3 跨管独立样本：两类预测展示，真值待标注 JPG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_3_Independent_transfer_two_classes.jpg)（4.24 MB）
- [三张关键图 合并PDF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Three_key_figures_two_class.pdf)（6.05 MB）
- [全部两类评价指标 Excel](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Three_key_figures_two_class_metrics.xlsx)（0.02 MB）
- [图稿、数据、源码、可重算真值与内部预测 主包](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Three_two_class_key_figures_complete.zip)（76.39 MB）
- [442张跨管两类原生预测标签 独立包](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Independent_442_two_class_predictions.zip)（27.76 MB）
- [三图与中英文结果、英文图注 Word](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Three_key_figures_captions_and_results.docx)（8.39 MB）
- [全部7位置两类预测对比图集 PDF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Independent_all_7_positions_two_class_atlas.pdf)（8.66 MB）
- [图1 时序基线：两类内部重新评价 PDF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_1_Temporal_baselines_two_classes.pdf)（0.07 MB）
- [图1 时序基线：两类内部重新评价 SVG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_1_Temporal_baselines_two_classes.svg)（0.06 MB）
- [图1 时序基线：两类内部重新评价 TIF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_1_Temporal_baselines_two_classes.tif)（1.50 MB）
- [图1 时序基线：两类内部重新评价 预览PNG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_1_Temporal_baselines_two_classes_preview.png)（0.14 MB）
- [图2 模块消融：两类评分与配对结构解码 PDF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_2_Module_ablations_two_classes.pdf)（0.07 MB）
- [图2 模块消融：两类评分与配对结构解码 SVG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_2_Module_ablations_two_classes.svg)（0.05 MB）
- [图2 模块消融：两类评分与配对结构解码 TIF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_2_Module_ablations_two_classes.tif)（1.14 MB）
- [图2 模块消融：两类评分与配对结构解码 预览PNG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_2_Module_ablations_two_classes_preview.png)（0.12 MB）
- [图3 跨管独立样本：两类预测展示，真值待标注 PDF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_3_Independent_transfer_two_classes.pdf)（5.92 MB）
- [图3 跨管独立样本：两类预测展示，真值待标注 SVG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_3_Independent_transfer_two_classes.svg)（8.84 MB）
- [图3 跨管独立样本：两类预测展示，真值待标注 TIF](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_3_Independent_transfer_two_classes.tif)（15.71 MB）
- [图3 跨管独立样本：两类预测展示，真值待标注 预览PNG](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Figure_3_Independent_transfer_two_classes_preview.png)（0.96 MB）
- [Final_verification.json](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Final_verification.json)（0.00 MB）
- [Independent_annotation_guide.txt](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Independent_annotation_guide.txt)（0.00 MB）
- [README.txt](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/README.txt)（0.00 MB）
- [结果与限制 中文文本](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/Results_and_limitations_CN.txt)（0.00 MB）
- [two_class_experiment_policy.json](https://raw.githubusercontent.com/Annalee811/LI/3839df9feaa49ecd30ffeec5f1d6593d205686d5/two_class_experiment_policy.json)（0.00 MB）

## 复现与独立标注

主包可重绘三张图并从缓存真值/预测重算26组混淆矩阵。独立标签包含442张两类掩膜：390张重训练头输出和52张固定参照；两包解压到同一父目录即可恢复完整资料。15份原始权重校验值未变，原模型、标注、旧结果均保留。

独立GT需从RGB标注背景0、既有1、新生2、不确定255；新生相对于前一个实际观测时期，已有根的延伸仍属既有。此前仅有“新生／非新生”的二值标签不足以区分既有根与背景。未完成人工标注或全255空模板会被评价入口拒绝。独立评价按整管汇总，包含人工有效背景；其分母与内部固定根前景评价不同。
