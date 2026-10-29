# V90、人工标注与原始 U-Net 对比
本包只评估与导出，未训练、未修改任何预测、人工标注或根系轮廓。原始 U-Net 为 V66（100轮，验证集宏平均 Dice 选择第37轮），不是 V75 加入 CG 后的版本。V90 为已发布的优化结果，包含你确认的 T2 根 0002-N1 人工复核修正，不能作为独立模型泛化精度。
人工参照固定为 V87：保留你授权的两处 T2 延伸改新生，以及全部 T4 延伸改新生；没有为本次比较再次改标签。人工标注是参照，不作为准确率100%的第三个模型。
## 指标
两输出使用相同人工标注、相同像素范围。以下是 T2–T4 时序分类，T1 仅基线展示。全部有效标注区包括训练区；原内部空间审计区也已用于反复反馈开发。均为同位置标注一致度，不是独立测试精度。指标限定在冻结的预测根系前景，不含前景之外的分割漏检。
| 范围 | 版本 | 像素准确率 | 持续 Dice | 新生 Dice | 延伸 Dice | 宏平均 Dice |
|---|---|---:|---:|---:|---:|---:|
| all_valid_annotated_root_pixels | original_U_Net_V66 | 0.939259 | 0.975512 | 0.851788 | 0.312411 | 0.713237 |
| all_valid_annotated_root_pixels | V90_reviewed_output | 0.961004 | 0.982058 | 0.907603 | 0.545661 | 0.811774 |
| original_inner_spatial_audit | original_U_Net_V66 | 0.924892 | 0.968708 | 0.818142 | 0.099134 | 0.628661 |
| original_inner_spatial_audit | V90_reviewed_output | 0.954844 | 0.977363 | 0.889470 | 0.435415 | 0.767416 |

CSV/JSON 同时给出各期及 T2–T4 汇总的 Dice、Precision、Recall、IoU、TP/FP/FN、类别支持数与混淆矩阵。T4 人工延伸已全部改新生，故延伸类无参照支持：有延伸误报时 Dice=0，双方都无该类时 Dice 未定义；另列仅参照中存在类别的宏平均。

## 原始三个模型的100轮曲线
曲线来自与原始 U-Net 同批的 V66：U-Net、DeepLabV3+、SegFormer-B0；冻结二值分割与特征后训练时序分类头，使用原始 (18).json。训练和验证损失直接取真实逐轮 history.csv，完整展示1–100轮，无平滑、插值、拼接或重算。原始 epoch0 未训练评估保留在原始CSV中，不算第1轮。选中轮次分别37、14、63，选择依据为验证宏平均Dice，而非最低验证损失。该历史曲线不是最新版人工标注的重新训练，也不是 V90 的训练曲线。

## 图像
图1为四期全部可用上半观察区，含RGB、人工标注、原始U-Net、V90；各分类列同时显示RGB叠加与白底根系。另有四张原生像素单期图。图2为四期同坐标局部，图3–4为六处此前反馈的细节，非根据Dice选取。人工标注列完整保留原标注范围，不伪造标注覆盖。分类图仅用最近邻缩放，TIFF为600dpi、黑色粗体Times New Roman。

## 下载
- [Fig1_Manual_Original_UNet_V90_four_period_full_comparison.tif](figures/Fig1_Manual_Original_UNet_V90_four_period_full_comparison.tif)
- [Full_T1_native_resolution_comparison.tif](figures/Full_T1_native_resolution_comparison.tif)
- [Full_T2_native_resolution_comparison.tif](figures/Full_T2_native_resolution_comparison.tif)
- [Full_T3_native_resolution_comparison.tif](figures/Full_T3_native_resolution_comparison.tif)
- [Full_T4_native_resolution_comparison.tif](figures/Full_T4_native_resolution_comparison.tif)
- [Fig2_Matched_four_period_local_manual_Original_UNet_V90.tif](figures/Fig2_Matched_four_period_local_manual_Original_UNet_V90.tif)
- [Fig3_Local_optimization_details_page_1.tif](figures/Fig3_Local_optimization_details_page_1.tif)
- [Fig4_Local_optimization_details_page_2.tif](figures/Fig4_Local_optimization_details_page_2.tif)
- [Loss_unet_complete_100_epochs.tif](figures/Loss_unet_complete_100_epochs.tif)
- [Loss_deeplab_complete_100_epochs.tif](figures/Loss_deeplab_complete_100_epochs.tif)
- [Loss_segformer_complete_100_epochs.tif](figures/Loss_segformer_complete_100_epochs.tif)
- [Fig5_Original_three_models_complete_100_epoch_training_validation_loss.tif](figures/Fig5_Original_three_models_complete_100_epoch_training_validation_loss.tif)
- [Fig6_Manual_Original_UNet_V90_accuracy_Dice_table.tif](figures/Fig6_Manual_Original_UNet_V90_accuracy_Dice_table.tif)
- [指标汇总 CSV](Accuracy_and_Dice_summary.csv)
- [完整100轮损失 CSV](Three_original_models_actual_100_epoch_losses.csv)
- [曲线来源与检查点记录](training_curve_provenance.json)

## 逐期结果的限制

逐期退化说明：全部有效标注区 T4 新生根 Dice 为原始 U-Net 0.824105、V90 0.717449，像素准确率为0.988346、0.984794。汇总提高不代表每期、每类都提高；T4 原人工延伸已改新生，而 V90 仍保留部分延伸预测。本报告如实保留该退化，不修改结果。
