"""Stage result-only assets and a download index in the existing upload checkout."""
from pathlib import Path
import argparse,hashlib,json,shutil

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--series',required=True)
parser.add_argument('--upload-checkout',type=Path,default=Path('/workspace/LI-registration-upload'))
args=parser.parse_args()
root=Path.cwd()/'results/microtube_registration';upload=args.upload_checkout
summary_dir=root/f'{args.series}_excluding_selected_series_results'
summary=json.loads((summary_dir/f'{args.series}_series_summary.json').read_text())
dest=upload/'roma-results'/args.series;dest.mkdir(parents=True,exist_ok=True)
names=['01_registered_sequence_composite.png','02_cumulative_transform_composite.png','03_cumulative_confidence_composite.png','04_overlay_50pct_composite.png','05_checkerboard_composite.png','06_effective_overlap_mask_composite.png']
labels=['配准原色RGB拼接','累计变换场预览','累计置信度','50%叠加','棋盘格质控','严格有效掩膜']

def copy(a,b):
 assert a.stat().st_size<100*1024**2,a
 b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
 assert hashlib.file_digest(a.open('rb'),'sha256').digest()==hashlib.file_digest(b.open('rb'),'sha256').digest()

for row in summary['positions']:
 pos=row['position'];src=root/f'{pos}_final_continuous_evaluation';out=dest/pos
 for f in src.iterdir():
  if f.is_file() and f.suffix in ['.png','.csv','.pdf','.json','.h5'] and f.name!='validation.json':copy(f,out/f.name)
 for suffix in ['continuous_registration_results.zip','full_T1_canvas_RGB_and_masks.zip','strict_common_effective_RGB.zip']:
  f=root.parent/f'{pos}_{suffix}';copy(f,out/f.name)
for f in summary_dir.iterdir():copy(f,dest/f.name)
for name in ['run_root_priority_series.py','evaluate_registration_series.py','package_registration_series.py','evaluate_continuous_registration.py','evaluate_adjacent_registration.py','finalize_continuous_registration.py','stage_registration_series.py']:
 copy(root/name,upload/'analysis-scripts'/name)

lines=[f'# {args.series} 四期连续配准结果','',
'处理位置：'+ '、'.join(r['position'] for r in summary['positions'])+'。以原始 T1 为参考，串联 T1→T2→T3→T4；保存原始RGB单次采样结果。','',
'每份结果包包含四期共同几何横向矩形的独立 RGB 与拼接图、严格共同有效区域掩膜、六类综合图、连续与相邻自动评估。严格有效区域透明 RGB、完整 T1 画布 RGB 和数值稠密变换场分别下载。','',
'| 位置 | 结果包 | 六类总览 | 全画布 RGB | 数值变换场 | 四期严格有效面积/T1 |','|---|---|---|---|---|---:|']
for r in summary['positions']:
 p=r['position'];lines.append(f"| {p} | [ZIP]({p}/{p}_continuous_registration_results.zip) | [PNG]({p}/continuous_registration_overview.png) | [ZIP]({p}/{p}_full_T1_canvas_RGB_and_masks.zip) | [HDF5]({p}/T1-T4_continuous_dense_fields.h5) | {r['common_effective_pct_T1']:.2f}% |")
lines+=['',f'[四期裁剪 RGB 汇总预览]({args.series}_series_cropped_RGB_preview.png) · [评估汇总 CSV]({args.series}_series_summary.csv)','']
positions=[r['position'] for r in summary['positions']]
lines+=['| 六类单独查看 | '+' | '.join(positions)+' |','|---|'+'---|'*len(positions)]
for label,name in zip(labels,names):lines.append('| '+label+' | '+' | '.join(f'[PNG]({p}/{name})' for p in positions)+' |')
lines+=['','严格有效区域透明 RGB 下载：'+' · '.join(f'[{p}]({p}/{p}_strict_common_effective_RGB.zip)' for p in positions),'',
'共同几何裁剪表示四期均有来源像素，不代表所有像素可靠。严格有效掩膜还要求双向置信度、循环一致性及正 Jacobian。根系细长结构候选用于优先粗匹配，土壤辅助；候选并非人工确认根系。','',
'| 位置 | 相邻自动质控 | 完整连续链自动质控 | 裁剪矩形 | 矩形内有效比例 |','|---|---:|---|---|---:|']
for r in summary['positions']:
 status='通过' if r['full_chain_auto_qc_pass'] else '未通过'
 lines.append(f"| {r['position']} | {r['adjacent_auto_qc_passed_pairs']}/3 | {status} | {r['crop_width']}×{r['crop_height']} | {r['effective_pct_inside_crop']:.2f}% |")
lines+=['','自动特征误差不是人工标志点 TRE，也并非保留验证集；未提供人工标志点及物理像素标定。自动质控通过不代表人工确认配准成功；低置信度和无有效掩膜区域不适合定量追踪。']
(dest/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Staged and hash-verified',len(list(dest.rglob('*'))),'result paths')
