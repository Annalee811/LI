# 最新：A017-2-3 共同区域复核

[查看定位、原色裁剪与下载](multiregion-review/README.md) · [下载分区结果](multiregion-review/A017-2-3_reviewed_regions_and_RGB.zip) · [下载独立拼接RGB](multiregion-review/A017-2-3_available_period_stitched_RGB.zip)

保存上缘A、中部B、下缘C；狭窄D候选未建立核验图，未裁剪、未计入面积。T3通过T4结构关系桥接定位，不能称为直接T1/T3或T2/T3独立验证成功。缓存RoMa T2/T3和本次额外T1/T4均为0个内部双向支持候选；对应位置依赖结构匹配与跨期桥接，不能称这些段直接神经匹配成功。以下旧结果仅作历史比较。

---

# 最新：同一区域定位与统一尺寸RGB

[定位图与独立裁剪](same-region-localization/README.md) · [下载图片](same-region-localization/A017-2-3_same_region_map_and_RGB.zip)

T1、T2、T3、T4，裁剪尺寸2517×512像素。四期齐全。区域定位仍以原始T1为标准；结构关系使用T1/T4、T3/T4、T2/T4。其中T2/T3的直接RoMa可靠点不足；粗区域假设来自明确记录的跨期结构关系，不能称为该段神经逐点配准成功。输出标为候选定位，尚未通过完整路线一致性核对，需人工确认。额外RoMa区域模型与最终定位的差异中位数/95百分位（T1像素）：T2/T4 14.16/32.34（不是人工误差）。未参与构建的结构路线对共同范围的差异中位数/95百分位（T1像素）：T1/T2 8.08/19.24。路线未达到5像素核对规则：T1_to_T2，须重点目视复核。
