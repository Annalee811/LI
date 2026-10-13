"""Stage only the B2 shape-preserving revision; preserves previous published outputs."""
from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parent;F=ROOT/'B2_shape_revision/final';D=json.loads((F/'evaluation.json').read_text());R=Path('/workspace/LI-registration-upload');P=Path('/workspace/LI/results');T=R/'roma-results/A017-1/A017-1-B2/shape-preserving-refinement';T.mkdir(exist_ok=True)
files=[F/'root_detail_comparison.png',F/'root_detail_preview.jpg',F/'root_structure_overview.png',F/'cropped_RGB_montage.png',F/'cropped_RGB_preview.jpg',F/'six_results_comprehensive.png',F/'six_results_preview.jpg',F/'thin_root_candidate_sites.png',F/'evaluation.json',F/'result_summary.md',F/'common_geometric_mask_T1.png',F/'common_effective_mask_T1.png']
files+=[F/f'T{t}_common_original_RGB.png' for t in D['period_ids']]
files+=sorted(P.glob('A017-1-B2_*fields.zip'))+sorted(P.glob('A017-1-B2_shape_preserving_*.zip'))
for p in files:
 dest=T/p.name;shutil.copy2(p,dest);assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
shutil.copytree(F/'scripts',T/'scripts',dirs_exist_ok=True)
readme='''# A017-1-B2 最新重新匹配：保持根系形状

T1→T2→T4，T3缺失。依据用户标示的横根、斜根及分叉重新定位；修正旧版重复土壤带错配。RGB只使用等比缩放、整体平移和小角度旋转，裁剪后保存原色，不使用RoMa非刚性变形。RoMa在三个1000像素重叠窗口独立细匹配，采用预设查询区域，不择最大置信度。

- [根系细节对照](root_detail_comparison.png)
- [三期裁剪原图拼接](cropped_RGB_montage.png)
- [六类综合质控图](six_results_comprehensive.png)
- [完整结果下载](A017-1-B2_shape_preserving_results.zip)
- [完整T1画布RGB下载](A017-1-B2_shape_preserving_full_RGB.zip)
- [保持形状的坐标场](A017-1-B2_global_coordinate_fields.zip)
- [T1→T2 RoMa候选稠密场](A017-1-B2_T1-T2_RoMa_candidate_fields.zip)
- [T2→T4 RoMa候选稠密场](A017-1-B2_T2-T4_RoMa_candidate_fields.zip)
- [评估摘要](result_summary.md)／[精确指标](evaluation.json)

独立无标注PNG：[T1](T1_common_original_RGB.png)、[T2](T2_common_original_RGB.png)、[T4](T4_common_original_RGB.png)。每幅2307×702 T1像素，600 DPI元数据。DPI及插值不会创造新采集细节。

自动误差为特征共识位点内的空间分块交叉验证，不是人工标志点TRE。候选根形支持不等同于根系分割、跨期根身份或新生/死亡判定。六类图中的RGB变换场是整体相似变换；单独的RoMa数值场明确标记为候选诊断，未用于RGB形变。

旧版仅保留作历史诊断，以本修订为当前查看版本。低置信度和超出共同视野的区域不能强行认定为连续根段。
'''
(T/'README.md').write_text(readme)
parent=T.parent/'README.md';s=parent.read_text() if parent.exists() else '# A017-1-B2\n\n';notice='## B2 根系细节重新匹配（保持形状）\n\n请优先查看[最新修订及下载](shape-preserving-refinement/README.md)。已修正重复土壤带错配；旧版含未经验证的非刚性RGB展示，仅留作历史诊断。\n\n'
if '## B2 根系细节重新匹配（保持形状）' not in s:parent.write_text(notice+s)
series=T.parent.parent/'README.md';s=series.read_text() if series.exists() else '# A017-1\n\n';notice='## B2 最新修订\n\n[根系指导、保持形状的重新匹配及下载](A017-1-B2/shape-preserving-refinement/README.md)。T3缺失，按T1→T2→T4处理。\n\n'
if '## B2 最新修订' not in s:series.write_text(notice+s)
manifest={str(p.relative_to(T)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in T.rglob('*') if p.is_file() and p.name!='SHA256.json'};(T/'SHA256.json').write_text(json.dumps(manifest,indent=2))
for p in T.rglob('*'):
 if p.is_file():assert p.stat().st_size<100*1024**2,(p,p.stat().st_size)
print('STAGED',T,len(manifest),'files',flush=True)
