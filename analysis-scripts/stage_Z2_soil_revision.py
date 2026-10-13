"""Publishable result assets for Z2 soil-shape refinement; preserves the failed baseline."""
from pathlib import Path
import shutil,hashlib,json
R=Path.cwd()/'results/microtube_registration';P=R/'A017-1-Z2_final_continuous_evaluation';C=R/'A017-1-Z2_soil_structure_coarse_results'
U=Path('/workspace/LI-registration-upload');D=U/'roma-results/A017-1/A017-1-Z2/soil-shape-refinement';D.mkdir(parents=True,exist_ok=True)
def copy(a,b):
 assert a.stat().st_size<100*1024**2,a
 b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
 assert hashlib.file_digest(a.open('rb'),'sha256').digest()==hashlib.file_digest(b.open('rb'),'sha256').digest()
for f in P.iterdir():
 if f.is_file() and f.suffix in ['.png','.csv','.json','.pdf','.txt','.h5'] and f.name!='validation.json':copy(f,D/f.name)
for f in C.iterdir():
 if f.is_file():copy(f,D/f.name if f.name=='T2_to_T4_local_RoMa_fields.h5' else D/'soil_structure_alignment'/f.name)
for name in ['A017-1-Z2_soil_shapes_revision_results.zip','A017-1-Z2_soil_revision_full_T1_RGB_and_masks.zip']:copy(R.parent/name,D/name)
for name in ['localize_Z2_soil_shapes.py','refine_Z2_soil_shapes.py','make_Z2_soil_focus_results.py','package_Z2_soil_revision.py','stage_Z2_soil_revision.py']:copy(R/name,U/'analysis-scripts'/name)
(D/'README.md').write_text('''# A017-1-Z2 土壤裂缝结构定位修订

按用户指出的下部裂缝、多边形边缘进行定位。实际时期为 T1→T2→T4，T3缺失。

[A/B/C 三期局部对照](soil_structure_alignment/soil_structure_window_comparison.png) · [结构定位全景图](soil_structure_alignment/soil_structure_focus_figure.png)

[全部图像结果 ZIP](A017-1-Z2_soil_shapes_revision_results.zip) · [完整 T1 画布 RGB 与掩膜 ZIP](A017-1-Z2_soil_revision_full_T1_RGB_and_masks.zip)

三期土壤结构粗对齐图、独立原色裁剪 RGB 与观察窗口为结构定位结果。A–C并非人工标志点。T1/T4、T2/T4的土壤描述子各有37个候选内点；拟合残差不等于独立标志点误差。

RoMa在共享区域中推断，保持目标原始画布尺寸，ROI外置信度为0。T2→T4严格有效面积从0提高到原始T4画布的3.43%；T1→T2的大区域和土壤局部重试仍少于接受条件要求的可靠匹配。完整链共同严格有效面积仍为0，未通过质控，六类连续综合图仅供诊断。

[六类连续 RoMa 综合图（诊断）](continuous_registration_overview.png) · [T2→T4局部配准RGB](T2_to_T4_local_registered_RGB.png) · [局部有效掩膜](T2_to_T4_local_effective_mask.png)

[T2→T4局部RoMa数值场 HDF5](T2_to_T4_local_RoMa_fields.h5) · [完整链诊断数值场 HDF5](T1-T2-T4_continuous_dense_fields.h5)

[土壤粗对齐仿射场 ROI HDF5](soil_structure_alignment/soil_structure_affine_fields_ROI.h5) 单独标为粗定位场，不具有RoMa置信度；不能用它替代通过有效性检查的稠密变换场。

原色RGB均从原始图单次采样；放大窗口只用于显示。未提供人工标志点，人工TRE未评估。不能据此对完整三期根系变化进行定量追踪。
''',encoding='utf-8')
parent=U/'roma-results/A017-1/README.md';old=parent.read_text();heading='## Z2 土壤结构定位修订'
if heading not in old:
 parent.write_text(heading+'\n\n[查看新的A/B/C结构对照与修订结果](A017-1-Z2/soil-shape-refinement/README.md)。T2→T4局部RoMa有效面积提高到3.43%，完整链仍未通过质控。原失败基线保留于下方结果。\n\n'+old,encoding='utf-8')
print('PASS hash-verified revision staging',len(list(D.rglob('*'))),'paths')
