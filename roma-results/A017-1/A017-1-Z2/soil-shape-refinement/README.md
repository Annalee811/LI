# A017-1-Z2 土壤裂缝结构定位修订

按用户指出的下部裂缝、多边形边缘进行定位。实际时期为 T1→T2→T4，T3缺失。

[A/B/C 三期局部对照](soil_structure_alignment/soil_structure_window_comparison.png) · [结构定位全景图](soil_structure_alignment/soil_structure_focus_figure.png)

[全部图像结果 ZIP](A017-1-Z2_soil_shapes_revision_results.zip) · [完整 T1 画布 RGB 与掩膜 ZIP](A017-1-Z2_soil_revision_full_T1_RGB_and_masks.zip)

三期土壤结构粗对齐图、独立原色裁剪 RGB 与观察窗口为结构定位结果。A–C并非人工标志点。T1/T4、T2/T4的土壤描述子各有37个候选内点；拟合残差不等于独立标志点误差。

RoMa在共享区域中推断，保持目标原始画布尺寸，ROI外置信度为0。T2→T4严格有效面积从0提高到原始T4画布的3.43%；T1→T2的大区域和土壤局部重试仍少于接受条件要求的可靠匹配。完整链共同严格有效面积仍为0，未通过质控，六类连续综合图仅供诊断。

[六类连续 RoMa 综合图（诊断）](continuous_registration_overview.png) · [T2→T4局部配准RGB](T2_to_T4_local_registered_RGB.png) · [局部有效掩膜](T2_to_T4_local_effective_mask.png)

[T2→T4局部RoMa数值场 HDF5](T2_to_T4_local_RoMa_fields.h5) · [完整链诊断数值场 HDF5](T1-T2-T4_continuous_dense_fields.h5)

[土壤粗对齐仿射场 ROI HDF5](soil_structure_alignment/soil_structure_affine_fields_ROI.h5) 单独标为粗定位场，不具有RoMa置信度；不能用它替代通过有效性检查的稠密变换场。

原色RGB均从原始图单次采样；放大窗口只用于显示。未提供人工标志点，人工TRE未评估。不能据此对完整三期根系变化进行定量追踪。
