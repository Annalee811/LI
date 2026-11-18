# V90 时序分割结果下载

**18 个拍摄位置，66 幅图像，来自8个微根管。** 所有结果保留当前配准RGB的原始像素尺寸。缺失时期没有补图。

[18页图像预览 PDF](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/All_positions_temporal_RGB_preview.pdf) · [指标与待核验清单 Excel](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_prediction_index_and_validation.xlsx) · [部署代码和权重 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_transfer_candidate_model.zip)

各位置文件包括：原背景三类标记图、类别掩膜、不确定区域图、带不确定编码的类别掩膜、白背景根系图、独占像素归属与推断时序ID。类别为持续根、新出现根、延伸段；第一期单独作为基准。不确定像素编码为255。

**模型版本说明：** 本批次是使用原V90冻结权重的部署适配版候选输出，每个位置独立构图。它重建了小段划分与邻接关系，近似了一个几何特征，并省略了未完整提供的原局部稠密预处理；因此不能视为完整复现原V90前处理或等同性能的最终模型。方法差异见 [输入适配检查记录](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/Input_adapter_audit.json)。

人工标注作为评价真值，不作为新图像预测输入。其他位置没有独立人工时序真值，未填写Dice、IoU等准确率；模型内部置信度不等于准确率。B3评价为同一开发位置上的参考一致性，T2–T4汇总于有效已分类真值根像素，背景及未分类/忽略像素不计入。

B3 另外提供 [原V90自动输出图包](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/B3_V90_automatic_outputs.zip)，与本批次适配版结果分开保存。

## 按拍摄位置下载

| 系列 | 拍摄位置 | 实际时期 | 图片包 | 大小 |
|---|---|---|---|---|
| 0353 | 0353-1-B3 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_0353-1-B3.zip) | 28.7 MiB |
| 0353 | 0353-2-B3 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_0353-2-B3.zip) | 37.9 MiB |
| 0353 | 0353-2-LB2 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_0353-2-LB2.zip) | 57.1 MiB |
| 0353 | 0353-2-LB3 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_0353-2-LB3.zip) | 59.1 MiB |
| 0353 | 0353-3-LB2 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_0353-3-LB2.zip) | 24.3 MiB |
| 0353 | 0353-3-LB3 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_0353-3-LB3.zip) | 47.7 MiB |
| A017 | A017-1-B2 | T1, T2, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_A017-1-B2.zip) | 40.1 MiB |
| A017 | A017-1-Z2 | T1, T2, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_A017-1-Z2.zip) | 12.5 MiB |
| A017 | A017-2-2 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_A017-2-2.zip) | 38.8 MiB |
| A017 | A017-2-3 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_A017-2-3.zip) | 44.6 MiB |
| A017 | A017-3-B3 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_A017-3-B3.zip) | 53.2 MiB |
| A017 | A017-3-Z2 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_A017-3-Z2.zip) | 56.2 MiB |
| Q1 | Q1-1-B2 | T1, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_Q1-1-B2.zip) | 14.4 MiB |
| Q1 | Q1-1-Z2 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_Q1-1-Z2.zip) | 22.4 MiB |
| Q1 | Q1-1-Z3 | T1, T2, T3 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_Q1-1-Z3.zip) | 33.7 MiB |
| Q1 | Q1-2-B3 | T1, T2, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_Q1-2-B3.zip) | 61.3 MiB |
| Q1 | Q1-2-Z2 | T1, T3, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_Q1-2-Z2.zip) | 46.9 MiB |
| Q1 | Q1-2-Z3 | T1, T2, T4 | [下载 ZIP](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_candidate_Q1-2-Z3.zip) | 55.6 MiB |

## 一次下载全部文件

下载并解压 [V90_download_index.zip](V90_download_index.zip)，在该目录运行：

```bash
python download_all.py --output ./V90_downloads
```

脚本会下载全部图片包、PDF、指标表、模型和脚本，并校验SHA256。下载中断后可重新运行；已校验文件会跳过。

[完整批处理脚本与日志](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/V90_batch_scripts_and_logs.zip) · [66幅图像清单](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/All_66_frames_status.csv) · [18位置清单](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/All_18_sequences_status.csv) · [训练资料与源代码索引](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/Training_data_and_source_index.md) · [输出完整性检查](https://raw.githubusercontent.com/Annalee811/LI/14a79d0f39809861e1ca50678cc9a3da64ecfdfb/Output_integrity_verification.json)

Q1-1-Z2 的 T2/T3 定位仍为粗配准、尚待核验，该位置相关时期额外标为不确定；Q1-3-Z3 未纳入本批次。

结果文件版本：`14a79d0f39809861e1ca50678cc9a3da64ecfdfb`。原始图像、已有标注、原权重和旧结果未覆盖。
