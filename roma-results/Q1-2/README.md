# Q1-2区域观察结果

- [图片包1：5张PNG](Q1-2_region_observation_images_part_1.zip)
- [图片包2：8张PNG](Q1-2_region_observation_images_part_2.zip)

各图片包为独立ZIP，只含PNG；有共同范围者保存T1尺度的同尺寸裁剪，无法定位者保留未确认标记与原图。

| 位置 | 实际时期 | 状态 | 定位/核对图 | 本位置图片 |
|---|---|---|---|---|
| Q1-2-B3 | T1/T2/T3/T4 | partial_neural_region_candidate | [PNG](Q1-2-B3/region-observation-results/Q1-2-B3_localization_review.png) | [ZIP](Q1-2-B3/region-observation-results/Q1-2-B3_partial_localization_review_and_RGB.zip) |
| Q1-2-Z2 | T1/T3/T4 | candidate_region_localization | [PNG](Q1-2-Z2/region-observation-results/Q1-2-Z2_same_region_localization.png) | [ZIP](Q1-2-Z2/region-observation-results/Q1-2-Z2_same_region_map_and_RGB.zip) |
| Q1-2-Z3 | T1/T2/T4 | region_localization | [PNG](Q1-2-Z3/region-observation-results/Q1-2-Z3_same_region_localization.png) | [ZIP](Q1-2-Z3/region-observation-results/Q1-2-Z3_same_region_map_and_RGB.zip) |

**Q1-2-B3**：前三期仅为RoMa模型候选；T4尚未定位. Failure remains failure. Missing periods are not synthesized; unlocalized existing periods are not relabeled missing. Only existing supported neural hypotheses are rendered as explicitly unverified candidates. Original unsupported RGB remains native, uncropped and unregistered. No elastic deformation or color changes.

**Q1-2-Z2**：缺少T2，不补造缺失图像。以原始T1为坐标与尺度基准，结构关系使用T1/T4、T3/T4。没有额外可靠结构路线可供核对，标为候选定位；不能据此声称精确逐点配准成功。

**Q1-2-Z3**：缺少T3，不补造缺失图像。以原始T1为坐标与尺度基准，结构关系使用T1/T4、T2/T4。额外结构路线在共同范围的差异中位数/95百分位（T1像素）：T1/T2 3.99/5.26。
