# A017-1 连续配准结果

处理位置：A017-1-B2、A017-1-Z2。以原始 T1 为参考，串联可用时期；保存原始RGB单次采样结果。

每份结果包包含可用时期共同几何横向矩形的独立 RGB 与拼接图、严格共同有效区域掩膜、六类综合图、连续与图像对自动评估。严格有效区域透明 RGB、完整 T1 画布 RGB 和数值稠密变换场分别下载。

| 位置 | 结果包 | 六类总览 | 全画布 RGB | 数值变换场 | 共同严格有效面积/T1 |
|---|---|---|---|---|---:|
| A017-1-B2 | [ZIP](A017-1-B2/A017-1-B2_continuous_registration_results.zip) | [PNG](A017-1-B2/continuous_registration_overview.png) | [ZIP](A017-1-B2/A017-1-B2_full_T1_canvas_RGB_and_masks.zip) | [HDF5](A017-1-B2/T1-T2-T4_continuous_dense_fields.h5) | 2.31% |
| A017-1-Z2 | [ZIP](A017-1-Z2/A017-1-Z2_continuous_registration_results.zip) | [PNG](A017-1-Z2/continuous_registration_overview.png) | [ZIP](A017-1-Z2/A017-1-Z2_full_T1_canvas_RGB_and_masks.zip) | [HDF5](A017-1-Z2/T1-T2-T4_continuous_dense_fields.h5) | 0.00% |

[裁剪 RGB 汇总预览](A017-1_series_cropped_RGB_preview.png) · [评估汇总 CSV](A017-1_series_summary.csv)

| 六类单独查看 | A017-1-B2 | A017-1-Z2 |
|---|---|---|
| 配准原色RGB拼接 | [PNG](A017-1-B2/01_registered_sequence_composite.png) | [PNG](A017-1-Z2/01_registered_sequence_composite.png) |
| 累计变换场预览 | [PNG](A017-1-B2/02_cumulative_transform_composite.png) | [PNG](A017-1-Z2/02_cumulative_transform_composite.png) |
| 累计置信度 | [PNG](A017-1-B2/03_cumulative_confidence_composite.png) | [PNG](A017-1-Z2/03_cumulative_confidence_composite.png) |
| 50%叠加 | [PNG](A017-1-B2/04_overlay_50pct_composite.png) | [PNG](A017-1-Z2/04_overlay_50pct_composite.png) |
| 棋盘格质控 | [PNG](A017-1-B2/05_checkerboard_composite.png) | [PNG](A017-1-Z2/05_checkerboard_composite.png) |
| 严格有效掩膜 | [PNG](A017-1-B2/06_effective_overlap_mask_composite.png) | [PNG](A017-1-Z2/06_effective_overlap_mask_composite.png) |

严格有效区域透明 RGB 下载：[A017-1-B2](A017-1-B2/A017-1-B2_strict_common_effective_RGB.zip) · [A017-1-Z2](A017-1-Z2/A017-1-Z2_strict_common_effective_RGB.zip)

共同几何裁剪表示所有可用时期均有来源像素，不代表所有像素可靠。严格有效掩膜还要求双向置信度、循环一致性及正 Jacobian。根系细长结构候选用于优先粗匹配，土壤辅助；候选并非人工确认根系。

| 位置 | 实际链 | 缺失时期 | 图像对自动质控 | 完整连续链自动质控 | 裁剪矩形 | 矩形内有效比例 |
|---|---|---|---:|---|---|---:|
| A017-1-B2 | T1→T2→T4 | T3 | 1/2 | 未通过 | 2303×402 | 14.43% |
| A017-1-Z2 | T1→T2→T4 | T3 | 0/2 | 未通过 | 1636×797 | 0.00% |

**A017-1-Z2：共同有效区域为 0，连续配准失败；RGB、变换场和其他可视化均仅为模型诊断结果，不能作为有效追踪依据。**

跨缺失期直接配准；不补造缺失图像，不能评估中间根系变化。自动特征误差不是人工标志点 TRE，也并非保留验证集；未提供人工标志点及物理像素标定。自动质控通过不代表人工确认配准成功；低置信度和无有效掩膜区域不适合定量追踪。
