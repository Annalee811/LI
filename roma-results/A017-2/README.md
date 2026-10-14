# 最新：A017-2 四期共同区域复核

[查看全系列定位总览](A017-2_reviewed_regions_overview.png) · [下载八张独立四期拼接RGB和总览](A017-2_all_stitched_RGB_and_overview.zip)

[下载当前全部分区结果图片](A017-2_all_same_region_images.zip)

原色图以各位置原始T1为坐标与尺度标准，每段只整体等比定位，保留根系和土层局部变化，不改色、不弹性扭曲。分段图24像素白带是非数据分隔，单段图无人工接缝。几何面积含明确标记的候选，不是可靠逐点匹配或根系匹配成功率。

| 位置 | 保留区域数 | T1几何覆盖（含候选） | 核验状态 | 定位图 | 完整分区结果 |
|---|---:|---:|---|---|---|
| A017-2-2 | 3 | 57.79% | B为候选 | [查看](A017-2-2/multiregion-review/A017-2-2_reviewed_region_localization.png) | [下载](A017-2-2/multiregion-review/A017-2-2_reviewed_regions_and_RGB.zip) |
| A017-2-3 | 3 | 68.32% | 通过可用区域路线核验；T3桥接 | [查看](A017-2-3/multiregion-review/A017-2-3_reviewed_region_localization.png) | [下载](A017-2-3/multiregion-review/A017-2-3_reviewed_regions_and_RGB.zip) |

保存上缘A、中部B、下缘C；B缺少额外达标路线，明确为候选。狭窄D候选未建立核验图，未裁剪、未计入面积。

保存上缘A、中部B、下缘C；狭窄D候选未建立核验图，未裁剪、未计入面积。T3通过T4结构关系桥接定位，不能称为直接T1/T3或T2/T3独立验证成功。缓存RoMa T2/T3和本次额外T1/T4均为0个内部双向支持候选；对应位置依赖结构匹配与跨期桥接，不能称这些段直接神经匹配成功。

自动区域路线和空间交叉验证依赖已选特征点及先验，不是人工TRE或完整连续逐点成功。根身份、新生和死亡未评估。所有失败与原始候选保留，不能通过扭曲图像消除质控分歧。下列旧记录仅作历史比较。

---

# 最新：A017-2同一区域定位与原色RGB

[下载本系列全部图片](A017-2_all_same_region_images.zip)

各位置使用自己的原始T1坐标与尺度，保留颜色与季节变化，无局部弹性变形。

| 位置 | 实际时期 | 独立裁剪尺寸 | 定位图 | 图片包 |
|---|---|---:|---|---|
| A017-2-2 | T1/T2/T3/T4 | 2102×558 | [PNG](A017-2-2/same-region-localization/A017-2-2_same_region_localization.png) | [ZIP](A017-2-2/same-region-localization/A017-2-2_same_region_map_and_RGB.zip) |
| A017-2-3 | T1/T2/T3/T4 | 2517×512 | [PNG](A017-2-3/same-region-localization/A017-2-3_same_region_localization.png) | [ZIP](A017-2-3/same-region-localization/A017-2-3_same_region_map_and_RGB.zip) |

**A017-2-2**：四期齐全。区域定位仍以原始T1为标准；结构关系使用T1/T4、T3/T4、T1/T2。未参与构建的结构路线对共同范围的差异中位数/95百分位（T1像素）：T1/T3 1.90/4.34；T2/T3 1.47/3.04；T2/T4 3.93/8.96。

**A017-2-3**：四期齐全。区域定位仍以原始T1为标准；结构关系使用T1/T4、T3/T4、T2/T4。其中T2/T3的直接RoMa可靠点不足；粗区域假设来自明确记录的跨期结构关系，不能称为该段神经逐点配准成功。输出标为候选定位，尚未通过完整路线一致性核对，需人工确认。额外RoMa区域模型与最终定位的差异中位数/95百分位（T1像素）：T2/T4 14.16/32.34（不是人工误差）。未参与构建的结构路线对共同范围的差异中位数/95百分位（T1像素）：T1/T2 8.08/19.24。路线未达到5像素核对规则：T1_to_T2，须重点目视复核。
