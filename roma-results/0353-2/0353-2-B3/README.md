# 最新：60张根系待复核标注，可直接修改

[下载原图、标注与可编辑页面](segmentation-pilot-annotation/reviewable-annotations/0353-2-B3_reviewable_annotations_v1.zip) · [使用说明](segmentation-pilot-annotation/reviewable-annotations/README.md)

打开包内 annotate.html 即显示60张现有标注。尚未经人员逐图复核，分类和粗细均可修改；没有真实模型评估指标。

---

# 最新：根系分割人工标注预实验准备

[下载60张原色图、人工标注工具及评估脚本](segmentation-pilot-annotation/0353-2-B3_human_annotation_kit_v4.zip) · [查看标注窗位置](segmentation-pilot-annotation/views/0353-2-B3_annotation_windows.png) · [标注规则](segmentation-pilot-annotation/annotation_protocol.md)

快捷键与标注说明版v4页面内置60张原色图，打开即显示；支持其他裁剪图原始比例预览，未匹配的图片不再静默过滤。固定实验图块经SHA核验后允许标注。支持Ctrl+Z/⌘Z撤销，并提供T1基线及完整标注范围说明。新增分类颜色按钮及颜色/时序/存活属性层查看；保留非互斥属性，导出科学标签编码不变。

以本位置四期分区裁剪RGB为输入。训练36、验证12、空间留出测试12图块，分别对应9、3、3物理观察窗；四期同窗同组、跨组缓冲64像素。单位置留出，不是独立位置测试。当前人工真值0、真实训练模型0，所有分割及根长一致性指标未测，未预填Dice0.956。

人工工具支持细笔刷、不确定区域、褐变、时序和疑似/确认死亡的独立属性；可见死亡根仍属于根系。确认死亡需记录依据，不能仅把褐变或一次未观察到判为死亡。独立人工根长需真实描线测量，无标尺时以T1像素计。

U-Net和DeepLabV3+软件数据流程已验证；RootPainter官方格式导出已核对，官方训练与预测仍需实际执行，不以普通U-Net改名替代。浏览器及合成软件检查不代表B3科学评估通过。标注工具是离线下载，无需上传图像。

以下为已保存的配准与裁剪结果。

---

# 最新：0353-2-B3 多段共同区域复核

[查看定位、根系对照和原色裁剪](multiregion-review/README.md) · [下载分区结果](multiregion-review/0353-2-B3_reviewed_regions_and_RGB.zip) · [下载四期拼接RGB](multiregion-review/0353-2-B3_four_stitched_RGB.zip)

原单段结果遗漏其他对应范围；本次几何覆盖80.82%（含候选），核验状态见结果页。下列旧结果仅供历史比较。

---

# 最新结果：同一区域定位与四期统一尺寸RGB裁剪

[定位图与独立裁剪图](same-region-localization/README.md) · [下载5张结果图](same-region-localization/0353-2-B3_same_region_map_and_four_RGB.zip)

四期裁剪尺寸：2541×738像素。三组相邻对应的自动点误差中位数均低于3个T1像素；另外三条独立区域路线差异中位数为1.07、2.57、3.80个T1像素。四期取景上下偏移较大，因此共同横向区域较窄。
