# 0353-3 四期连续配准结果

处理位置：0353-3-LB2、0353-3-LB3。以原始 T1 为参考，串联 T1→T2→T3→T4；保存原始RGB单次采样结果。

每份结果包包含四期共同几何横向矩形的独立 RGB 与拼接图、严格共同有效区域掩膜、六类综合图、连续与相邻自动评估。严格有效区域透明 RGB、完整 T1 画布 RGB 和数值稠密变换场分别下载。

| 位置 | 结果包 | 六类总览 | 全画布 RGB | 数值变换场 | 四期严格有效面积/T1 |
|---|---|---|---|---|---:|
| 0353-3-LB2 | [ZIP](0353-3-LB2/0353-3-LB2_continuous_registration_results.zip) | [PNG](0353-3-LB2/continuous_registration_overview.png) | [ZIP](0353-3-LB2/0353-3-LB2_full_T1_canvas_RGB_and_masks.zip) | [HDF5](0353-3-LB2/T1-T4_continuous_dense_fields.h5) | 5.09% |
| 0353-3-LB3 | [ZIP](0353-3-LB3/0353-3-LB3_continuous_registration_results.zip) | [PNG](0353-3-LB3/continuous_registration_overview.png) | [ZIP](0353-3-LB3/0353-3-LB3_full_T1_canvas_RGB_and_masks.zip) | [HDF5](0353-3-LB3/T1-T4_continuous_dense_fields.h5) | 2.37% |

[四期裁剪 RGB 汇总预览](0353-3_series_cropped_RGB_preview.png) · [评估汇总 CSV](0353-3_series_summary.csv)

| 六类单独查看 | 0353-3-LB2 | 0353-3-LB3 |
|---|---|---|
| 配准原色RGB拼接 | [PNG](0353-3-LB2/01_registered_sequence_composite.png) | [PNG](0353-3-LB3/01_registered_sequence_composite.png) |
| 累计变换场预览 | [PNG](0353-3-LB2/02_cumulative_transform_composite.png) | [PNG](0353-3-LB3/02_cumulative_transform_composite.png) |
| 累计置信度 | [PNG](0353-3-LB2/03_cumulative_confidence_composite.png) | [PNG](0353-3-LB3/03_cumulative_confidence_composite.png) |
| 50%叠加 | [PNG](0353-3-LB2/04_overlay_50pct_composite.png) | [PNG](0353-3-LB3/04_overlay_50pct_composite.png) |
| 棋盘格质控 | [PNG](0353-3-LB2/05_checkerboard_composite.png) | [PNG](0353-3-LB3/05_checkerboard_composite.png) |
| 严格有效掩膜 | [PNG](0353-3-LB2/06_effective_overlap_mask_composite.png) | [PNG](0353-3-LB3/06_effective_overlap_mask_composite.png) |

严格有效区域透明 RGB 下载：[0353-3-LB2](0353-3-LB2/0353-3-LB2_strict_common_effective_RGB.zip) · [0353-3-LB3](0353-3-LB3/0353-3-LB3_strict_common_effective_RGB.zip)

共同几何裁剪表示四期均有来源像素，不代表所有像素可靠。严格有效掩膜还要求双向置信度、循环一致性及正 Jacobian。根系细长结构候选用于优先粗匹配，土壤辅助；候选并非人工确认根系。

| 位置 | 相邻自动质控 | 完整连续链自动质控 | 裁剪矩形 | 矩形内有效比例 |
|---|---:|---|---|---:|
| 0353-3-LB2 | 2/3 | 未通过 | 2523×821 | 14.24% |
| 0353-3-LB3 | 2/3 | 未通过 | 2534×818 | 6.61% |

自动特征误差不是人工标志点 TRE，也并非保留验证集；未提供人工标志点及物理像素标定。自动质控通过不代表人工确认配准成功；低置信度和无有效掩膜区域不适合定量追踪。
