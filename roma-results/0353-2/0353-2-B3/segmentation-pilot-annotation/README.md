# 当前：人工T1/T2锚定的连续预测 v3

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
