# 最新：Q1 同一区域分区复核

[查看全系列定位总览](Q1_reviewed_regions_overview.png) · [下载三个已定位位置的九张独立拼接RGB及五位置总览](Q1_all_located_stitched_RGB_and_overview.zip)

Q1-1-B2和Q1-2-Z2分别补回不同偏移的第二个区域，Q1-2-Z3保留一个共同范围；未定位的位置明确保留失败与候选。仅两期的Q1-1-B3按要求跳过。统一尺寸仅针对同一位置、同一观察区域，不替换原图清晰度或宣称生成真实新细节。

| 位置 | 有图时期 | 本次状态 | 定位/核对图 | 独立RGB与结果图片 |
|---|---|---|---|---|
| Q1-1-B2 | T1, T3, T4 | 通过本次可用区域路线规则；非人工确认 | [查看](../Q1-1/Q1-1-B2/multiregion-review/Q1-1-B2_reviewed_region_localization.png) | [下载](../Q1-1/Q1-1-B2/multiregion-review/Q1-1-B2_reviewed_regions_and_RGB.zip) |
| Q1-2-B3 | T1, T2, T3, T4 | 前三期结构定位候选；T4未定位 | [查看](../Q1-2/Q1-2-B3/multiregion-review/Q1-2-B3_localization_review.png) | [下载](../Q1-2/Q1-2-B3/multiregion-review/Q1-2-B3_partial_localization_review_and_RGB.zip) |
| Q1-2-Z2 | T1, T3, T4 | 通过本次可用区域路线规则；非人工确认 | [查看](../Q1-2/Q1-2-Z2/multiregion-review/Q1-2-Z2_reviewed_region_localization.png) | [下载](../Q1-2/Q1-2-Z2/multiregion-review/Q1-2-Z2_reviewed_regions_and_RGB.zip) |
| Q1-2-Z3 | T1, T2, T4 | 通过本次可用区域路线规则；非人工确认 | [查看](../Q1-2/Q1-2-Z3/multiregion-review/Q1-2-Z3_reviewed_region_localization.png) | [下载](../Q1-2/Q1-2-Z3/multiregion-review/Q1-2-Z3_reviewed_regions_and_RGB.zip) |
| Q1-3-Z3 | T1, T2, T3 | T1未定位；原图核对，无共同裁剪 | [查看](../Q1-3/Q1-3-Z3/multiregion-review/Q1-3-Z3_localization_review.png) | [下载](../Q1-3/Q1-3-Z3/multiregion-review/Q1-3-Z3_partial_localization_review_and_RGB.zip) |

缺少T2；保存上部A和下部B两段，下部T1对应后期上部。以原始1330像素宽T1为尺度，不通过放大宣称新增细节。

T1/T2/T3更新为局部结构定位候选，T4原图存在但尚未定位。T2/T3只有15个稳定结构锚点，额外核验路线不足，不能称为四期共同区或精准逐点匹配。

缺少T2；保存上部A和下部B两段，分别采用不同T1/T3、T3/T4偏移。下部点级误差较大，区域定位核验不能当作精准根系逐点重合。

缺少T3；保存一个共同观察范围。区域路线满足规则，仍保留根系和土层局部移动。

缺少T4；新完整画布结构搜索仍未定位T1。保留三期原始RGB核对图，未猜测共同裁剪坐标。

多区域拼接白带为非数据分隔，未观察的间隙不能用于根系连续性结论。几何覆盖不是可靠点或根系匹配比例；区域核验依赖描述子和先验，不是人工TRE。原始RGB仅经每区整体等比定位及单次采样，原色、无弹性扭曲。候选图和未定位原图不能冒充完整连续配准。所有脚本、源图SHA、RoMa候选、失败及核验证据独立保留。下列旧记录仅作历史比较。

---

# Q1系列区域观察与未确认核对图

[查看五个位置的总览](Q1_all_region_observation_overview.png)

- [Q1-1图片包1：4张PNG](../Q1-1/Q1-1_region_observation_images.zip)
- [Q1-2图片包2：5张PNG](../Q1-2/Q1-2_region_observation_images_part_1.zip)
- [Q1-2图片包3：8张PNG](../Q1-2/Q1-2_region_observation_images_part_2.zip)
- [Q1-3图片包4：4张PNG](../Q1-3/Q1-3_region_observation_images.zip)

5个位置保存结果，1个位置仅两期按要求跳过。总21张主结果PNG：5张定位/核对图、16张独立RGB。独立RGB中包含未定位的原图，不能称为全部已配准或全部具有相同裁剪尺寸。Q1-2-B3仅前三期模型候选，T4未定位；Q1-3-Z3未建立T1共同区域；Q1-2-Z2为缺少额外路线的候选。

| 位置 | 状态 | 定位/核对图 | 独立图片包 |
|---|---|---|---|
| Q1-1-B2 | 共同区域定位 | [PNG](../Q1-1/Q1-1-B2/region-observation-results/Q1-1-B2_same_region_localization.png) | [ZIP](../Q1-1/Q1-1-B2/region-observation-results/Q1-1-B2_same_region_map_and_RGB.zip) |
| Q1-2-B3 | 前三期模型候选；T4未定位 | [PNG](../Q1-2/Q1-2-B3/region-observation-results/Q1-2-B3_localization_review.png) | [ZIP](../Q1-2/Q1-2-B3/region-observation-results/Q1-2-B3_partial_localization_review_and_RGB.zip) |
| Q1-2-Z2 | 候选区域定位，需复核 | [PNG](../Q1-2/Q1-2-Z2/region-observation-results/Q1-2-Z2_same_region_localization.png) | [ZIP](../Q1-2/Q1-2-Z2/region-observation-results/Q1-2-Z2_same_region_map_and_RGB.zip) |
| Q1-2-Z3 | 共同区域定位 | [PNG](../Q1-2/Q1-2-Z3/region-observation-results/Q1-2-Z3_same_region_localization.png) | [ZIP](../Q1-2/Q1-2-Z3/region-observation-results/Q1-2-Z3_same_region_map_and_RGB.zip) |
| Q1-3-Z3 | T1未定位；原图核对，无共同裁剪 | [PNG](../Q1-3/Q1-3-Z3/region-observation-results/Q1-3-Z3_localization_review.png) | [ZIP](../Q1-3/Q1-3-Z3/region-observation-results/Q1-3-Z3_partial_localization_review_and_RGB.zip) |

Q1-1-B3只有T3/T4，已跳过；缺失时期不补造，未定位时期不伪装为缺失。图片ZIP只含PNG，脚本与全部失败诊断独立保存。原色、无局部弹性变形，真实人工TRE未评估。
