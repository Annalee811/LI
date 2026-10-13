"""Finalize B2 shape-preserving outputs, detailed QC, faithful metrics and portable archives."""
from pathlib import Path
import json,csv,hashlib,zipfile,shutil
import cv2,numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
import evaluate_continuous_registration as v
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'B2_shape_revision';F=OUT/'final';D=json.loads((F/'evaluation.json').read_text());TS=D['period_ids'];RAW=ROOT/'input/raw/V-变化/A017-1-B2';x0,y0,x1,y1=D['common_crop_xyxy'];rw,rh=x1-x0,y1-y0
font=ImageFont.truetype(v.REG,26);small=ImageFont.truetype(v.REG,21)
# Compact detailed comparison. A shared physical window uses the exact same display scale in all three columns.
rowheights=[];patches=[]
for window in D['observation_windows']:
 label=window['label'];images=[Image.open(F/f'T{t}_detail_{label}_RGB.png') for t in TS];w,h=images[0].size;scale=min(1.,650/w,270/h);size=(round(w*scale),round(h*scale));patches.append([im.resize(size,Image.Resampling.LANCZOS) for im in images]);rowheights.append(size[1]+60)
canvas=Image.new('RGB',(2200,sum(rowheights)+90),'white');draw=ImageDraw.Draw(canvas)
for k,t in enumerate(TS):draw.text((185+k*670,12),f'T{t}：原色根系细节',font=font,fill='#22333c')
y=75
for window,images,height in zip(D['observation_windows'],patches,rowheights):
 draw.text((12,y+10),window['label'],font=font,fill='#27687d');draw.text((12,y+45),window['description'],font=small,fill='#334455')
 for k,im in enumerate(images):canvas.paste(im,(185+k*670,y));draw.line((185+k*670,y+height-22,835+k*670,y+height-22),fill='#d4dce0',width=1)
 y+=height
canvas.save(F/'root_detail_comparison.png',dpi=(600,600));canvas.resize((1600,round(canvas.height*1600/canvas.width)),Image.Resampling.LANCZOS).save(F/'root_detail_preview.jpg',quality=96,subsampling=0)
# Assess native target support. Outside the evaluated shared crop is UNKNOWN, not low confidence.
for met in D['roma_evaluation']:
 a,b=[int(z[1:]) for z in met['pair'].split('_to_')];p=F/met['pair'];H=np.array(D['matrices_T1_to_native'][f'T{b}']);Ha=np.array(D['matrices_T1_to_native'][f'T{a}']);w,h=Image.open(RAW/f'T{b}.jpg').size;sw,sh=Image.open(RAW/f'T{a}.jpg').size
 cropH=H@np.array([[1,0,x0],[0,1,y0],[0,0,1]])
 with h5py.File(p/'RoMa_candidate_fields.h5','r') as f:valid=f['effective_for_global_RGB'][:];conf=f['confidence_target_query'][:]
 assessed=cv2.warpPerspective(np.ones((rh,rw),np.uint8),cropH,(w,h),flags=cv2.INTER_NEAREST)>0
 validnative=cv2.warpPerspective(valid.astype('uint8'),cropH,(w,h),flags=cv2.INTER_NEAREST)>0
 lownative=cv2.warpPerspective((conf<.5).astype('uint8'),cropH,(w,h),flags=cv2.INTER_NEAREST)>0
 geometric=cv2.warpPerspective(np.ones((sh,sw),np.uint8),H@np.linalg.inv(Ha),(w,h),flags=cv2.INTER_NEAREST)>0
 met.update(geometric_overlap_pct_native_target=100*geometric.mean(),assessed_crop_pct_native_target=100*assessed.mean(),effective_pct_native_target=100*validnative.mean(),low_confidence_pct_native_target=100*lownative.mean(),unknown_outside_assessed_pct_native_target=100*(~assessed).mean())
 Image.fromarray(validnative.astype('uint8')*255).save(p/'effective_mask_native_target.png');Image.fromarray(geometric.astype('uint8')*255).save(p/'geometric_mask_native_target.png')
 (p/'evaluation.json').write_text(json.dumps(met,indent=2))
# Six result classes, both actual pairs on a common T1 observation crop.
figs=[Image.open(F/name/'six_results_composite.png') for name in ['T1_to_T2','T2_to_T4']];six=Image.new('RGB',(sum(z.width for z in figs),max(z.height for z in figs)),'white');xx=0
for im in figs:six.paste(im,(xx,0));xx+=im.width
six.save(F/'six_results_comprehensive.png',dpi=(600,600));six.resize((1800,round(six.height*1800/six.width)),Image.Resampling.LANCZOS).save(F/'six_results_preview.jpg',quality=94,subsampling=0)
Image.open(F/'cropped_RGB_montage.png').resize((2100,round((rh+108)*2100/(rw*3))),Image.Resampling.LANCZOS).save(F/'cropped_RGB_preview.jpg',quality=97,subsampling=0)
# Thin, individual automatic candidate-site markers; no root identities asserted.
features={};view=Image.new('RGB',(rw*3,rh+110),'white');draw=ImageDraw.Draw(view)
for k,t in enumerate(TS):view.paste(Image.open(F/f'T{t}_common_original_RGB.png'),(k*rw,110));draw.text((k*rw+12,12),f'T{t}：自动根形支持位点（细线）',font=font,fill='#23343d')
counts=[]
for name in ['T1_to_T2','T2_to_T4']:
 a,b=[int(z[1:]) for z in name.split('_to_')];rows=list(csv.DictReader((F/f'{name}_feature_matches.csv').open()));count=0
 for row in rows:
  if row['root_support_proxy']!='1':continue
  pa=np.array([float(row['source_x_analysis']),float(row['source_y_analysis']),1.])*Image.open(RAW/f'T{a}.jpg').width/1600;pa[2]=1
  pb=np.array([float(row['target_x_analysis']),float(row['target_y_analysis']),1.])*Image.open(RAW/f'T{b}.jpg').width/1600;pb[2]=1
  pts={a:np.linalg.inv(np.array(D['matrices_T1_to_native'][f'T{a}']))@pa,b:np.linalg.inv(np.array(D['matrices_T1_to_native'][f'T{b}']))@pb}
  if not all(x0<=p[0]<x1 and y0<=p[1]<y1 for p in pts.values()):continue
  count+=1
  for t,p in pts.items():xx=TS.index(t)*rw+p[0]-x0;yy=p[1]-y0+110;draw.ellipse((xx-3,yy-3,xx+3,yy+3),outline='#b6e9e8',width=1)
 counts.append(dict(pair=name,displayed_root_support_sites=count))
view.save(F/'thin_root_candidate_sites.png',dpi=(600,600));D['thin_root_site_display']=counts
D['automatic_error_interpretation']='Spatial-block cross-validation within preselected descriptor consensus; conditional automatic verification, not independently annotated landmark TRE.'
D['input_sha256']={f'T{t}':hashlib.sha256((RAW/f'T{t}.jpg').read_bytes()).hexdigest() for t in TS}
(F/'evaluation.json').write_text(json.dumps(D,ensure_ascii=False,indent=2))
# Concise report uses actual measurements and clearly states denominator / missing T3.
lines=['# A017-1-B2 重新匹配结果','', 'T1→T2→T4；T3缺失。已修正原先重复土壤带的错误定位。图像只使用整体等比缩放、平移和小角度旋转，随后裁剪、拼接；不使用局部非刚性变形。原色RGB只从原图采样一次。','',f'共同最大横向矩形：{rw} × {rh} 个T1像素。完整T1共同几何面积：{D["common_geometric_pct_T1"]:.2f}%。', '', '| 配准对 | 去重特征匹配 | 根形支持候选 | 自动验证中位误差/T1像素 | 有效面积/共同裁剪 | 低置信度/共同裁剪 |', '|---|---:|---:|---:|---:|---:|']
for m in D['roma_evaluation']:
 fs=next(z for z in D['feature_evaluation'] if z['pair']==m['pair']);lines.append(f'| {m["pair"]} | {fs["unique_consensus_matches"]} | {fs["root_support_candidate_matches"]} | {fs["spatial_cross_validation_median_T1_px"]:.2f} | {m["effective_pct_common_canvas"]:.2f}% | {m["low_confidence_pct_common_canvas"]:.2f}% |')
lines+=['',f'自动图像对通过比例：{D["automatic_pair_success_ratio"]:.0%}（2组实际相邻可用时期）；不等同于根系识别或四期追踪成功率。连续共同有效区域占裁剪图 {D["common_effective_pct_crop"]:.2f}%，占完整T1 {D["common_effective_pct_T1"]:.2f}%。', '',f'链式变换与独立T1→T4拟合的位置差：中位数{D["chain_vs_independent_direct_median_T1_px"]:.2f}、P95 {D["chain_vs_independent_direct_p95_T1_px"]:.2f}个T1像素；属于内部一致性检验。','', '误差来自自动特征的空间分块交叉验证；没有人工标志点坐标，真实标志点误差未评估。有效掩膜要求RoMa双向置信度≥0.5、循环误差≤3个T1像素、候选场无折叠且与最终整体变换偏差≤3个T1像素。低置信度只统计实际评估区域，区域外记为未知。', '', 'A–H是依据截图提示选择的观察窗口。细线圆点仅表示自动匹配的根形支持候选；不表示所有真实根，不能据此判断新生或死亡。超出T4视野的分枝下端不强行补齐。T2原图宽1330像素，统一至T1尺度不能增加未采集的细节。','', '形状保持变换场与RoMa候选稠密场分别保存并标明类型。RoMa候选场不用于生成最终RGB。完整Python脚本、数值场和逐点CSV一并保留。']
(F/'result_summary.md').write_text('\n'.join(lines)+'\n')
# Verify original RGB and shape preservation again after all rendering refinements.
with h5py.File(F/'shape_preserving_coordinate_fields.h5','r') as fields:
 for t in TS:
  matrix=fields[f'T{t}/T1_to_native_matrix'][:];singular=np.linalg.svd(matrix[:2,:2],compute_uv=False);np.testing.assert_allclose(singular[0],singular[1],rtol=1e-10)
  source=np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB'));coords=fields[f'T{t}/T1_crop_to_native_xy_px'][:];got=cv2.remap(source,coords[...,0],coords[...,1],cv2.INTER_LINEAR);np.testing.assert_array_equal(got,np.array(Image.open(F/f'T{t}_common_original_RGB.png')))
print('PASS final RGB single resampling, original colours, equal singular values',flush=True)
# Also retain literal source-pixel crops with no interpolation, enhancement or rotation.
D['literal_native_crops']={}
for t in TS:
 im=Image.open(RAW/f'T{t}.jpg').convert('RGB');H=np.array(D['matrices_T1_to_native'][f'T{t}']);corners=np.array([[x0,y0,1],[x1-1,y0,1],[x1-1,y1-1,1],[x0,y1-1,1]])@H.T
 box=[max(0,int(np.floor(corners[:,0].min()))),max(0,int(np.floor(corners[:,1].min()))),min(im.width,int(np.ceil(corners[:,0].max()))+1),min(im.height,int(np.ceil(corners[:,1].max()))+1)]
 native_crop=im.crop(box);native_crop.save(F/f'T{t}_literal_native_original_crop.png',dpi=(600,600));np.testing.assert_array_equal(np.array(native_crop),np.array(im)[box[1]:box[3],box[0]:box[2]])
 D['literal_native_crops'][f'T{t}']=dict(native_bbox_xyxy=box,kind='Literal native RGB pixels, axis-aligned enclosing rectangle; different native sizes, not registered display')
(F/'evaluation.json').write_text(json.dumps(D,ensure_ascii=False,indent=2))
# Additional pair-only crops retain branches visible in T1/T2 but outside T4's FOV.
import register_roma as rr
D['pair_only_original_crops']=[]
base=rr.grid(D['T1_size_wh'][1],D['T1_size_wh'][0])
for a,b in [(1,2),(2,4)]:
 geometry=[]
 for t in [a,b]:
  native=rr.project(base,np.array(D['matrices_T1_to_native'][f'T{t}']));geometry.append(rr.inside(native,Image.open(RAW/f'T{t}.jpg').size))
 box=v.rectangle(np.logical_and.reduce(geometry));bx0,by0,bx1,by1=box;out=F/'adjacent_geometric_crops'/f'T{a}_to_T{b}';out.mkdir(parents=True,exist_ok=True);pims=[]
 for t in [a,b]:
  pim=Image.open(F/f'T{t}_full_T1_canvas_RGB.png').crop(box);pim.save(out/f'T{t}_pair_only_original_RGB.png',dpi=(600,600));pims.append(pim)
 plate=Image.new('RGB',(2*pims[0].width,pims[0].height+90),'white');pd=ImageDraw.Draw(plate)
 for k,(t,pim) in enumerate(zip([a,b],pims)):plate.paste(pim,(k*pim.width,90));pd.text((k*pim.width+15,12),f'T{t} 相邻可用时期共同原色 RGB',font=font,fill='#24343d')
 plate.save(out/'pair_only_original_RGB_montage.png',dpi=(600,600))
 D['pair_only_original_crops'].append(dict(pair=f'T{a}_to_T{b}',T1_xyxy=box,notes='Pair geometric crop within T1 FOV. Extra area outside three-period shared crop has no RoMa confidence assessment.'))
(F/'evaluation.json').write_text(json.dumps(D,ensure_ascii=False,indent=2))
# Per-window diagnostics: assess the precise root observation windows, without root identity claims.
window_rows=[]
for window in D['observation_windows']:
 bx0,by0,bx1,by1=window['T1_xyxy'];sl=np.s_[by0-y0:by1-y0,bx0-x0:bx1-x0]
 for m in D['roma_evaluation']:
  with h5py.File(F/m['pair']/'RoMa_candidate_fields.h5','r') as f:
   eff=f['effective_for_global_RGB'][sl];conf=f['confidence_target_query'][sl]
  window_rows.append(dict(label=window['label'],pair=m['pair'],description=window['description'],effective_pct=100*eff.mean(),low_confidence_pct=100*(conf<.5).mean()))
(F/'root_window_quality.json').write_text(json.dumps(window_rows,ensure_ascii=False,indent=2))
shutil.copy2(OUT/'whole_scene_validation_baseline.json',F/'whole_scene_validation_baseline.json')
# Reproducible scripts and exact candidate-detection evidence.
scripts=F/'scripts';scripts.mkdir(exist_ok=True)
for name in ['rematch_B2_shapes.py','build_B2_shape_preserving.py','finalize_B2_shape_revision.py','B2_tiled_roma.py','publish_B2_shape_revision.py','register_roma.py','evaluate_continuous_registration.py','run_0353_1_root_priority.py','evaluate_adjacent_registration.py']:shutil.copy2(ROOT/name,scripts/name)
shutil.copy2(OUT/'shape_feature_candidates.json',F/'shape_feature_candidates.json')
P=Path('/workspace/LI/results');archives={
 'A017-1-B2_shape_preserving_results.zip':[p for p in F.rglob('*') if p.is_file() and p.name!='six_results_composite.png' and p.suffix!='.h5' and 'full_T1_canvas' not in p.name and 'literal_native_original_crop' not in p.name and 'adjacent_geometric_crops' not in p.parts],
 'A017-1-B2_global_coordinate_fields.zip':[F/'shape_preserving_coordinate_fields.h5'],
 'A017-1-B2_T1-T2_RoMa_candidate_fields.zip':[F/'T1_to_T2/RoMa_candidate_fields.h5'],
 'A017-1-B2_T2-T4_RoMa_candidate_fields.zip':[F/'T2_to_T4/RoMa_candidate_fields.h5'],
 'A017-1-B2_shape_preserving_full_RGB.zip':[p for p in F.glob('*full_T1_canvas*')]+list(F.glob('*literal_native_original_crop*'))+list((F/'adjacent_geometric_crops').rglob('*.png'))+[F/'common_geometric_mask_T1.png',F/'common_effective_mask_T1.png']}
for name,files in archives.items():
 with zipfile.ZipFile(P/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:z.write(p,p.relative_to(F))
 with zipfile.ZipFile(P/name) as z:assert z.testzip() is None
 assert (P/name).stat().st_size<100*1024**2,(name,(P/name).stat().st_size)
 print('ARCHIVE',name,round((P/name).stat().st_size/1024**2,2),'MiB',flush=True)
print('FINAL',json.dumps({k:D[k] for k in ['common_effective_pct_T1','common_effective_pct_crop','automatic_pair_success_ratio','roma_evaluation']},ensure_ascii=False),flush=True)
