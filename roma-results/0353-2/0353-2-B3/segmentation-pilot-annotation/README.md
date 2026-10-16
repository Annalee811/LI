# 当前：原标注核对、三模型比较与四期预测

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
