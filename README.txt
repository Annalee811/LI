新生根单类结果包 / Newly observed roots only — 2026-11-23

这次是从已有模型输出中提取新生类并重算/导出新生Precision与Recall，未重新训练模型。
原多类模型、原始图像、人工标注、权重和旧结果均保留。这里只展示新生根，不把延伸段等其他类别合并为新生。
新生定义：原时序类别2。其他有效像素仍作为非新生参与误检、漏检统计，防止缩小分母。
Precision=TP/(TP+FP)，Recall=TP/(TP+FN)。无预测正像素时Precision未定义，不填0。

结果协议分别为：
1. B3全图历史人工一致性：T2–T4合并，包含人工有效背景，忽略255不确定像素。
2. 分区隔离内部模块对照：共同的有效留出根像素；固定上游前景，不包含背景。
3. 配对完整视野结构解码：同一检查点的相同神经输出，另行比较解码阶段。
请分别使用三类协议，不拼接为一条连续优化曲线，不把B3历史一致性称为跨管准确率。
既有检查点选择保持原样，本次没有按新指标重新选权重或改测试集。

主图：B3_V66_V90_new_roots_only_SCI.*（P、R及同视野T1–T4的RGB/人工/V66/V90）
实验图：New_root_precision_recall_SCI.*（新生P、R的内部控制与配对解码）
字体Times New Roman、黑字白底；位图600 dpi，PDF嵌入字体，SVG可编辑。
New_root_results_and_captions.docx：可编辑中英文结果及英文图注。
New_root_Precision_Recall.xlsx：只包含新生类P、R及其计数、清单，不输出其他类别指标。

All_18_series_new_roots_only.pdf：18个位置、66幅图像，每页原RGB/新生叠加/新生黑白掩膜。
每管ZIP里的每个位置均提供：
  Tn_new_mask.png：原生尺寸，0非新生、255新生。
  Tn_new_RGB_overlay.png：仅新生像素着色，其他原始RGB像素逐值不变。
  Tn_new_root_RGB_on_white.png：新生根的原RGB像素，其他像素白色。
  Tn_new_ownership.npz：仅保留新生位置的原axis/track_id，其他位置-1；不强制填补未知归属。
  New_root_T1-T4.jpg：该位置的四期合成页；未采集日期留空并标Not acquired。
T1是基准，空新生掩膜不代表根系没有发生或真实新生率为0。缺失时期不合成；变化参照前一个实际日期。
Q1-1-Z2的T2/T3为粗定位，按描述性预测展示，不进入主独立测试。Q1-3-Z3排除。

B3使用原自动V90输出；其他位置使用已记录、含近似的V90部署适配结果。
除B3外尚无人工时序真值，不计算跨微根管P或R，不将预测当人工真值。
基线/消融预测单类包另含442张新生二值掩膜（390张新训练输出、52张固定参照）。

未来独立评价：沿用固定3管、7位置、26幅输入，其中7个基准期不评新生，19个后续影像需人工真值。
独立输入名单保持不变。在清单中填写human_complete、annotator、annotation_source，标签来自RGB。
python evaluate_new_roots.py --manifest 完成清单.csv --annotation-root /path/to/GT  --prediction-dir /path/to/one/model/seed --output /new/empty/evaluation
默认GT为原时序标签0/1/2/3/255（2新生），预测为0/255新生掩膜。
也可用--gt-encoding new-binary接受Tn_new_GT.png：0非新生、1新生、255不确定。
空模板/无人工来源会拒绝评分。按整管汇总，不把日期或小图当作独立生物重复。

Source scripts include explicit source paths in the current environment. Override the module
constants W/STUDY/O/REF/RGBROOT/F if reproducing elsewhere. Standard dependencies:
numpy, Pillow, matplotlib, PyMuPDF, openpyxl, python-docx. No Torch is needed for this derivation.

随包重算新生指标（无需原机器路径或多类模型）：
python recompute_new_metrics_from_bundle.py --data-root . --output /new/empty/counts
随包二值真值、固定留出像素、内部预测及历史V66/V90新生掩膜可重算26组统计，计数应与表格一致。
