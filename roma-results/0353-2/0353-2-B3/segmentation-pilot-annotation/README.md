# 当前：三模型同轮优化与四类完整成果 v9

[下载本轮全部ZIP](three-model-continuity-v9/README.md)。U-Net、原上传DeepLabV3+、SegFormer-B0均执行8轮细根加权/连续性微调，再统一验证连接与小噪点剔除；仅按C3选择。最终验证Dice为0.7538 / 0.6573 / 0.7087，重复C4为0.6346 / 0.5206 / 0.5217。C4仅DeepLab提升，另两模型略降；保留真实负结果。U-Net微调未胜过起始网络，保留原网络权重并交付本轮选定后处理。

四类成果为论文图片、三模型四期分割与完整评估、三模型权重与完整脚本、训练切块与原始人工标注。全部原标注保持不变，四期同空间区域同组，独立外部测试图像为0；未进行时序类别或其他位置预测。

# 前期：根／背景细节优化与论文格式图片 v6

[新增：根系短断点与分叉连接优化 v8](root-continuity-v8/README.md)。以当前v7起始单模型输出为基线，从训练区人工连接关系学习轻量判别器；验证区选参后，重复C4的预测连通域265→227，人工根段参考点对连通覆盖率75.99%→77.24%，严格Dice0.6349→0.6356，提升有限。提供四期前后掩膜、所有补接细节和独立平滑中心线SVG；人工真值不变，显示平滑不计入指标，未宣称新的独立高精度模型。

[新增：清晰训练图块筛选对照 v7](clear-training-patches-v7/README.md)。120张训练候选窗口筛为70张根系窗口及4张背景窗口，验证与重复空间留出不筛选、人工标注不变。原采样与筛选采样均从同一权重继续训练20轮，均未超过起始模型，未替换已发布的v6集成。筛选数据、实际训练曲线和全部结果已保存。

[新增：三模型根长回归与一致性图片 ZIP](paper-refinement-v6/three-model-root-length/0353-2-B3_three_model_root_length_regression_v6.zip)：U-Net、DeepLabV3+、SegFormer-B0 均有单独回归图和同坐标综合图，验证与重复空间留出分开。

[下载论文图 ZIP](paper-refinement-v6/0353-2-B3_paper_figures_v6.zip) · [下载四期分割结果 ZIP](paper-refinement-v6/0353-2-B3_root_results_v6.zip) · [下载模型与脚本 ZIP](paper-refinement-v6/0353-2-B3_refined_models_v6.zip) · [下载384像素训练数据 ZIP](paper-refinement-v6/0353-2-B3_native384_training_data_v6.zip)

新增 ImageNet 预训练 U-Net、原像素384上下文、翻转平均和保护细根的保守去噪。按 C3 验证选择的等权 U-Net 集成，验证 Dice **0.7539**，重复空间留出 Dice **0.6227**、Precision **0.6340**、Recall **0.6118**、IoU **0.4521**；上一版原 U-Net 留出 Dice 为0.5718。原始预测和去噪结果均保留。人工标注及原始 RGB 不变。

六幅主图、两幅补充图统一白底／灰色／青绿色，提供600dpi PNG、LZW TIFF、PDF，定量图另有SVG。根长一致性 R²=0.8556、RMSE=155.97原像素，参考长度由人工笔刷掩膜骨架得到，并非独立人工测量。训练／验证／重复留出切块120／40／40张，同一区域四期同组。

**仍未达到近1指标：只有一个位置、四张原始时期图，C4已多次查看，独立外部测试图像为0，未启动时序分类或预测其他位置。** 新增清理参数只从既有 C3 网格选择，但在初次排名评估后追加；完整选择记录、像素级得失、原始排名和全部测得指标均已保存。图片采用论文格式不代表模型准确率已经达到发表要求。

[详细结果与全部下载](paper-refinement-v6/README.md)

# 先前：四期上部根／背景二分类 v5

[下载根／背景分割与评估](root-background-stage-v5/0353-2-B3_root_background_v5_results.zip) · [下载人工标注切块数据](root-background-stage-v5/0353-2-B3_root_background_v5_data.zip) · [下载三个模型与完整脚本](root-background-stage-v5/0353-2-B3_root_background_v5_models.zip)

使用最新四期上部人工标注，独立训练 U-Net、上传的原版 TensorFlow DeepLabV3+、SegFormer-B0 根／背景二分类。输入只有当前 RGB，保留全部原始时序属性，死亡不纳入。256×256原像素切块训练120、验证40、空间留出40张，同一位置的四期同组，跨组64像素隔离。

**本轮尚未达标：验证集选中 U-Net，验证 Dice 0.7399，空间留出 Dice 0.5718；留出 Precision 0.5771、Recall 0.5665、IoU 0.4003。因此不启动时序追踪。**

本阶段先检查根覆盖、背景误检和严格 Precision／Recall／Dice／IoU。只有验证及空间留出均达到根 Dice≥0.95、Precision≥0.90、Recall≥0.95 才允许进入后续时序阶段；本轮没有启动时序训练或预测其他位置。实际指标、门槛判定、根长一致性和原像素核对图见结果包。仅一个位置，独立外部测试图像为0。

[本轮详细说明与全部下载文件](root-background-stage-v5/README.md)

# 先前：人工T1/T2锚定的连续预测 v3

[下载连续图像与可编辑结果](sequential-anchored-v3/0353-2-B3_sequential_T1T2_manual_T3T4_results_v3.zip) · [下载新增模型与训练记录](sequential-anchored-v3/0353-2-B3_sequential_models_and_training_v3.zip)

保留用户T1/T2原人工标签；T3参考T2人工根，T4参考T3预测根。前期未见部分属于新增，保留新生与根尖延伸子类；消失候选单列，不能等同死亡。没有褐变/死亡监督标签，未训练这两类。只使用原上部区域。

新联合分割未胜过原U-Net，最终根分割继续使用原模型；新增均衡时序分支在同一T1人工参考协议下验证持续、新生、延伸Dice为0.7156、0.6367、0.2400，延伸识别仍弱。验证T2根Dice为0.7399，空间留出0.5493；不是跨位置独立验证，T3/T4类别未经定量验证。可在包内annotate.html选择images四张RGB并导入editable_upper_RGB_sequential.json继续修改。

# 先前：原标注核对、三模型比较与四期预测

[下载原始人工标注核对图](three-model-progress-v2/0353-2-B3_original_manual_label_review.zip) · [下载三模型训练与评估](three-model-progress-v2/0353-2-B3_three_models_training_v2.zip) · [下载四期预测](three-model-progress-v2/0353-2-B3_four_period_predictions_v2.zip)

原标注从用户进度JSON无损读取：T1蓝色基础根；T2紫色持续根69201像素、绿色新生根37360像素、黄色延伸根5824像素（指定上部区域）。原标注与模型预测分开保存，T1/T2原标签完整保留。可编辑进度中T3/T4全部为模型预测，忽略此前笔迹，不合并。

根／背景分割验证Dice：U-Net 0.7443，SegFormer-B0 0.6797，上传的DeepLabV3+ 0.6278，按验证集选择U-Net。T1/T2标注与上一轮一致，复用前两个已训练模型，新增SegFormer试训。另为选中架构训练时序分类模块；三类原标签均参与，但延伸根验证Dice为0，目前不能可靠识别。T3/T4类别准确率未评估。仅一个位置的空间留出预实验，不代表独立位置泛化。

# 先前：T1/T2 上部根系分割试训

[下载切块数据与两模型结果](trained-upper-t1t2-v1/0353-2-B3_upper_T1T2_data_and_results.zip) · [下载上传的DeepLabV3+模型权重](trained-upper-t1t2-v1/0353-2-B3_DeepLabV3plus_TF112_model.zip)

只用用户指定的上部2541×1108区域。原始256×256图块训练60、验证20、空间留出测试20；同区T1/T2同组，跨组至少64像素间隔。

本轮二分类试训的验证／空间留出Dice：U-Net 0.7443／0.5907；上传的TensorFlow 1.12 DeepLabV3+（ResNet-v2-50）0.6278／0.5005。指标针对现有人工掩膜，非跨位置验证。详细Precision、Recall、Dice、IoU、预测图、权重、来源校验及完整脚本均已保存。

# 当前：自选图片标注版

[下载离线标注工具 ZIP](custom-images-v1/root_annotation_custom_images_v1.zip)

解压后在 Chrome / Edge 打开 annotate.html，自行选择完整 RGB 或裁剪图即可标注。已取消原来60张图块的名称、哈希和448×448尺寸限制；没有预置图片。

保持原图尺寸的画笔与掩膜导出，最高60倍显示，工具栏固定在图旁，切图保留倍率与原图坐标。Ctrl+Z 撤销，Ctrl+S 下载全部标注进度 JSON，未完成也可保存。重新打开时选择相同原图，再导入本版 JSON 恢复。JSON 不含 RGB，须保留原图。

浏览器验证包含大尺寸和非正方形图片的实际指针绘制、撤销、跨期切换、同名不同内容隔离、进度恢复及原尺寸导出。验证仅涉及工具功能和数据完整性。

旧工具与旧图包保留在 legacy-zoom60-v1/、full-image-annotation-v1/ 和 two-window-review-v3/。
