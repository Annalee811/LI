# 新生根专版：精确率、召回率及全部系列图像

仅提取原时序类别2（新生），只报告Precision与Recall，不输出其他类别的性能指标，也不把其他类别并入新生。现有模型与人工标注保持原样，本次未重新训练模型。

B3全图历史一致性：V66新生Precision为85.03%、Recall为85.32%；V90为94.51%和84.50%。这些是T2–T4在B3人工有效全图像素（含背景）的描述性结果，不是独立微根管准确率。

18个位置、66幅原始RGB上均已导出仅新生的标记图。首期保留原图并标基准，未采集日期不补图。每管ZIP独立解压，包含所有位置的原生新生掩膜、原图叠加、根RGB提取及原根轴归属。B3来自原自动V90输出，其余位置来自已记录的V90部署适配预测。跨管没有人工真值，精确率和召回率均未计算。

基线/消融统计与配对结构解码另用各自内部协议展示；不能与B3全图历史分数混作一条优化曲线。误差为三个优化种子的SD，不是生物学重复。无预测正像素时Precision未定义，不填0。

## 下载

- [新生专版SCI对比图 JPG：新生精确率、召回率及四期同视野RGB标记](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/B3_V66_V90_new_roots_only_SCI.jpg)（5.93 MB）
- [全部18位置、66幅图像的新生根PDF（18页）](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/All_18_series_new_roots_only.pdf)（13.38 MB）
- [新生精确率、召回率全部统计 Excel](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_Precision_Recall.xlsx)（0.03 MB）
- [新生图稿、指标、源码及可独立重算的二值评价数据](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_figures_metrics_code.zip)（53.39 MB）
- [新生根结果分析与英文图注 Word](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_results_and_captions.docx)（11.82 MB）
- [66幅新生预测原生文件索引](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/All_66_new_root_prediction_manifest.csv)（0.04 MB）
- [新生专版SCI对比图 PDF](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/B3_V66_V90_new_roots_only_SCI.pdf)（9.53 MB）
- [新生专版SCI对比图 可编辑SVG](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/B3_V66_V90_new_roots_only_SCI.svg)（14.38 MB）
- [新生专版SCI对比图 TIFF（600 dpi）](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/B3_V66_V90_new_roots_only_SCI.tif)（21.82 MB）
- [图稿预览：B3_V66_V90_new_roots_only_SCI](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/B3_V66_V90_new_roots_only_SCI_preview.png)（1.36 MB）
- [基线/消融及固定参照的442张新生二值预测掩膜](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/Baseline_ablation_442_new_only_masks.zip)（5.71 MB）
- [独立样本的新生预测对比PDF（7页，不赋准确率）](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/Independent_new_roots_only_comparison.pdf)（8.14 MB）
- [单类导出完整性检查](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_only_final_checks.json)（0.00 MB）
- [内部基线与消融：仅新生精确率、召回率 JPG](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_precision_recall_SCI.jpg)（1.18 MB）
- [内部基线与消融：仅新生P/R PDF](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_precision_recall_SCI.pdf)（0.07 MB）
- [内部基线与消融：仅新生P/R 可编辑SVG](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_precision_recall_SCI.svg)（0.05 MB）
- [内部基线与消融：仅新生P/R TIFF](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_precision_recall_SCI.tif)（0.88 MB）
- [图稿预览：New_root_precision_recall_SCI](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_precision_recall_SCI_preview.png)（0.10 MB）
- [新生根结果分析 中文文本](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_root_results_CN.txt)（0.00 MB）
- [原生尺寸新生图包：微根管 0353-1](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_0353-1.zip)（15.03 MB）
- [原生尺寸新生图包：微根管 0353-2](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_0353-2.zip)（77.98 MB）
- [原生尺寸新生图包：微根管 0353-3](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_0353-3.zip)（37.12 MB）
- [原生尺寸新生图包：微根管 A017-1](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_A017-1.zip)（26.99 MB）
- [原生尺寸新生图包：微根管 A017-2](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_A017-2.zip)（42.82 MB）
- [原生尺寸新生图包：微根管 A017-3](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_A017-3.zip)（54.83 MB）
- [原生尺寸新生图包：微根管 Q1-1](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_Q1-1.zip)（36.74 MB）
- [原生尺寸新生图包：微根管 Q1-2](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/New_roots_native_Q1-2.zip)（83.29 MB）
- [完整说明](https://raw.githubusercontent.com/Annalee811/LI/35bf973d3d5a4e5bbcdaba3c94c286bf2d532b1c/README.txt)（0.00 MB）

## 使用说明

投稿用图为Times New Roman黑色字体、白色背景，600 dpi，并提供PDF与可编辑SVG。18页PDF的三行依次为原RGB、金色新生叠加、黑色新生掩膜；原生掩膜PNG编码为0非新生、255新生。首期空掩膜仅表示不评价新生，不能解释为新生率为0。缺失日期的变化相对于前一个实际观测时期。

完整图稿包包含可独立重算指标的二值真值与预测：解压后执行`python recompute_new_metrics_from_bundle.py --data-root . --output 新空目录`。26组统计的TP、FP、FN已重新计算并与新生表格逐一核对。

未来跨管评价沿用3管7位置26幅的固定输入清单，其中19幅非基准图像需人工新生真值。可直接标新生二类（0非新生、1新生、255不确定），无需为本终点标其他类别。未完成人工标注或全255模板均会被评分入口拒绝。

原多类权重、人工标注、图像和旧实验均保留。本分支为新增的新生专版，未覆盖旧分支。
