# Q1-2-Z2 区域观察结果

**状态：candidate_region_localization**

缺少T2，不补造缺失图像。以原始T1为坐标与尺度基准，结构关系使用T1/T4、T3/T4。没有额外可靠结构路线可供核对，标为候选定位；不能据此声称精确逐点配准成功。

各实际时期的独立裁剪图均为2493×1295像素。细框标出全部实际时期均有原始来源像素的最大横向矩形。原始RGB单次全局相似变换采样；T1裁剪像素与原图对应矩形相同，无局部弹性拉伸。

- [查看定位或未确认核对图](Q1-2-Z2_same_region_localization.png)
- 独立RGB：[T1_common_region_RGB.png](T1_common_region_RGB.png)／[T3_common_region_RGB.png](T3_common_region_RGB.png)／[T4_common_region_RGB.png](T4_common_region_RGB.png)
- [下载本位置全部图片](Q1-2-Z2_same_region_map_and_RGB.zip)

图片包仅包含4张PNG，不含过程或代码目录。保持原色，不作局部弹性变形；未定位原图不声称已配准，也不声称具有统一裁剪尺寸。自动区域对应与条件空间分块检查不等于人工真实TRE或逐根身份确认。没有人工标志点，真实TRE未评估。DPI或统一尺度不会增加未拍摄的细节。候选结果不能用于直接定量归因根的生长、消失或死亡。脚本与审计数据独立保存在analysis-scripts。
