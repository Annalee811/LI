#!/usr/bin/env python3
"""Evaluate the complete RoMa chain on native T1 coordinates; export six panels.
Run from the LI repository root. No RGB enhancement, feature-based refitting,
or extrapolated pixels. Full T1 canvas and a geometric-common rectangle are saved.
"""
from pathlib import Path
import sys, os, json, csv, gc, hashlib
os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

REPO=Path.cwd().resolve(); ROOT=REPO/'results/microtube_registration'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_adjacent_registration import independent_points,stats,save_csv
POSITION='0353-2-B3'
RAW=ROOT/'input/raw/V-变化'/POSITION; PAIR=ROOT/POSITION
OUT=ROOT/f'{POSITION}_final_continuous_evaluation';OUT.mkdir(parents=True,exist_ok=True)
FIELDS=OUT/'dense_fields';FIELDS.mkdir(exist_ok=True)
DETAIL=OUT/'native_T1_results';DETAIL.mkdir(exist_ok=True)
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
REG='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
THRESH=.5; CYCLE_T1=3.0
cv2.setNumThreads(4);cv2.setRNGSeed(20261012)


def rgb(p):
 with Image.open(p) as im:return np.array(im.convert('RGB'))


def grid(h,w):
 y,x=np.mgrid[:h,:w].astype(np.float32);return np.stack([x,y],-1)


def inside(m,wh):
 w,h=wh;return np.isfinite(m).all(-1)&(m[...,0]>=0)&(m[...,0]<=w-1)&(m[...,1]>=0)&(m[...,1]<=h-1)


def remap(a,m,nearest=False,border=-1):
 if m.shape[0]==0 or m.shape[1]==0:
  return np.empty(m.shape[:2]+a.shape[2:],dtype=a.dtype)
 return cv2.remap(a,m[...,0].astype(np.float32),m[...,1].astype(np.float32),cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=border)


def jac(m):
 xy,xx=np.gradient(m[...,0]);yy,yx=np.gradient(m[...,1]);return xx*yy-xy*yx


def save_png(name,a):
 Image.fromarray(a).save(OUT/name,dpi=(600,600),compress_level=5)


def save_native(name,a):
 Image.fromarray(a).save(DETAIL/name,dpi=(600,600),compress_level=5)


def load(n):
 with np.load(PAIR/n/'dense_transform.npz') as z:return {k:z[k] for k in z.files}


def rectangle(mask):
 """Exact largest-area axis-aligned rectangle with width >= height."""
 h,w=mask.shape; heights=np.zeros(w,np.int32);best=(0,0,0,0,0)
 for y,row in enumerate(mask):
  heights=np.where(row,heights+1,0);stack=[]
  for x in range(w+1):
   cur=int(heights[x]) if x<w else 0;start=x
   while stack and stack[-1][1]>cur:
    sx,hh=stack.pop();width=x-sx;start=sx
    if width>=hh and width*hh>best[0]:best=(width*hh,sx,y-hh+1,x,y+1)
   if not stack or stack[-1][1]<cur:stack.append((start,cur))
 if best[0]==0:raise ValueError('No common geometric horizontal rectangle')
 return list(best[1:])


def montage(images,labels,name,subtitle=''):
 h,w=images[0].shape[:2];header=108 if subtitle else 76
 canvas=Image.new('RGB',(len(images)*w,h+header),'white');d=ImageDraw.Draw(canvas)
 f=ImageFont.truetype(FONT,31);sf=ImageFont.truetype(REG,21)
 for i,(a,label) in enumerate(zip(images,labels)):
  canvas.paste(Image.fromarray(a),(i*w,header));d.text((i*w+18,12),label,font=f,fill=(26,32,38))
  if subtitle:d.text((i*w+18,60),subtitle,font=sf,fill=(85,93,100))
  d.line((i*w,header,i*w+w,header),fill=(190,197,202),width=2)
  d.rectangle((i*w,header,i*w+w-1,header+h-1),outline=(210,216,221),width=2)
 canvas.save(OUT/name,dpi=(600,600),compress_level=5)
 return canvas


def flow_preview(mapping,wh,base,geometry,scale):
 disp=(mapping+.5)*np.array([base.shape[1]/wh[0],base.shape[0]/wh[1]],np.float32)-.5-base
 mag,ang=cv2.cartToPolar(disp[...,0],disp[...,1]);hsv=np.empty((*mag.shape,3),np.uint8)
 hsv[...,0]=np.mod(ang*90/np.pi,180).astype(np.uint8);hsv[...,1]=255
 hsv[...,2]=np.rint(np.clip(np.log1p(mag)/np.log1p(scale),0,1)*255).astype(np.uint8)
 a=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB);a[~geometry]=(235,235,235)
 for y in range(72,a.shape[0]-48,160):
  for x in range(72,a.shape[1]-48,160):
   if geometry[y,x] and mag[y,x]>1:
    direction=disp[y,x]/mag[y,x];end=(int(round(x+36*direction[0])),int(round(y+36*direction[1])))
    cv2.arrowedLine(a,(x,y),end,(35,35,35),4,tipLength=.28)
    cv2.arrowedLine(a,(x,y),end,(250,250,250),2,tipLength=.28)
 return a


def configure_position(position):
 global POSITION,RAW,PAIR,OUT,FIELDS,DETAIL
 POSITION=position;RAW=ROOT/'input/raw/V-变化'/position;PAIR=ROOT/position
 OUT=ROOT/f'{position}_final_continuous_evaluation';OUT.mkdir(parents=True,exist_ok=True)
 FIELDS=OUT/'dense_fields';FIELDS.mkdir(exist_ok=True)
 DETAIL=OUT/'native_T1_results';DETAIL.mkdir(exist_ok=True)


def main():
 raw_paths=[next(f for f in RAW.iterdir() if f.stem.upper()==f'T{i}') for i in range(1,5)]
 sources=[rgb(n) for n in raw_paths]
 t1=sources[0];h,w=t1.shape[:2];base=grid(h,w)
 adj=[load(n) for n in ['T1_to_T2','T2_to_T3','T3_to_T4']]
 cumulative=base.copy();confidence=np.ones((h,w),np.float32)
 geometry=np.ones((h,w),bool);valid=np.ones((h,w),bool)
 maps=[];confs=[];geoms=[];valids=[];cycles=[];aligned=[t1];rows=[];evaluations=[]
 for k,f in enumerate(adj):
  previous=cumulative.copy();pwh=tuple(f['source_size_wh']);twh=tuple(f['target_size_wh'])
  step_source_conf=remap(f['confidence_source'],previous,border=0)
  next_map=remap(f['source_to_target_xy_px'],previous,border=-1)
  next_geo=geometry&inside(previous,pwh)&inside(next_map,twh)
  step_target_conf=remap(f['confidence_target'],next_map,border=0)
  confidence=np.minimum(confidence,np.minimum(step_source_conf,step_target_conf)).clip(0,1)
  confidence[~next_geo]=0
  propagated=remap(f['valid_target'].astype(np.uint8),next_map,nearest=True,border=0)>0
  return_map=next_map.copy()
  for z in adj[:k+1][::-1]:return_map=remap(z['target_to_source_xy_px'],return_map,border=-1)
  cycle=np.linalg.norm(return_map-base,axis=-1)
  positive=jac(next_map)>0
  valid=valid&next_geo&propagated&(confidence>=THRESH)&(cycle<=CYCLE_T1)&positive
  geometry=next_geo;cumulative=next_map
  maps.append(cumulative.copy());confs.append(confidence.copy());geoms.append(geometry.copy());valids.append(valid.copy());cycles.append(cycle.copy())
  im=remap(sources[k+1],cumulative,border=0);im[~geometry]=0;aligned.append(im)
  save_native(f'T{k+2}_registered_on_T1_RGB.png',im)
  save_native(f'T{k+2}_path_confidence_16bit.png',np.rint(confidence*65535).astype(np.uint16))
  save_native(f'T{k+2}_effective_mask.png',valid.astype(np.uint8)*255)
  save_native(f'T{k+2}_geometric_mask.png',geometry.astype(np.uint8)*255)
  save_native(f'T{k+2}_low_confidence_mask.png',(geometry&(confidence<THRESH)).astype(np.uint8)*255)
  np.savez_compressed(FIELDS/f'T1_to_T{k+2}_cumulative_on_T1.npz',
    T1_grid_to_period_xy_px=cumulative.astype(np.float32),confidence_path_min=confidence.astype(np.float32),
    effective_mask_T1=valid,geometric_mask_T1=geometry,cycle_error_T1_px=cycle.astype(np.float32),
    T1_size_wh=np.array([w,h]),period_size_wh=np.array(twh),confidence_threshold=np.float32(THRESH),
    cumulative_cycle_threshold_T1_px=np.float32(CYCLE_T1))
  meta=dict(source_file=str(raw_paths[0].relative_to(REPO)),target_file=str(raw_paths[k+1].relative_to(REPO)))
  pa,pb,mi,sb=independent_points(meta)
  pred_period=remap(cumulative,pa[None,...])[0]
  point_back=pb[None,...].astype(np.float32)
  backward_supported=np.ones(len(pb),bool)
  for z in adj[:k+1][::-1]:
   backward_supported &= inside(point_back,tuple(z['target_size_wh']))[0]
   point_back=remap(z['target_to_source_xy_px'],point_back,border=-1)
   backward_supported &= inside(point_back,tuple(z['source_size_wh']))[0]
  point_back=point_back[0]
  err_t1=np.linalg.norm(point_back-pa,axis=1)
  err_period=np.linalg.norm(pred_period-pb,axis=1)
  normalized=np.linalg.norm((pred_period-pb)/sb,axis=1)
  forward_supported=remap(geometry.astype(np.uint8),pa[None,...],True,border=0)[0]>0
  supported=forward_supported & backward_supported
  good=remap(valid.astype(np.uint8),pa[None,...],True,border=0)[0]>0
  good &= supported
  error_t1_stats=stats(err_t1[supported]);error_period_stats=stats(err_period[supported]);norm_stats=stats(normalized[supported])
  if len(pa):
   pr=[]
   for j in range(len(pa)):
    pr.append(dict(point_id=j+1,T1_x=float(pa[j,0]),T1_y=float(pa[j,1]),
      period_x=float(pb[j,0]),period_y=float(pb[j,1]),predicted_T1_x=float(point_back[j,0]),predicted_T1_y=float(point_back[j,1]),
      error_T1_px=float(err_t1[j]) if supported[j] else None,error_native_period_px=float(err_period[j]) if supported[j] else None,forward_path_available=bool(forward_supported[j]),backward_path_available=bool(backward_supported[j]),in_geometric_path=bool(supported[j]),in_effective_path=bool(good[j])))
   save_csv(OUT/f'T1_to_T{k+2}_automatic_endpoint_points.csv',pr)
  total=h*w;g=int(geometry.sum());e=int(valid.sum());low=int((geometry&(confidence<THRESH)).sum())
  cy_stats=stats(cycle[valid])
  checks=dict(effective_area_at_least_10pct=e/total>=.10,
       endpoint_validation_points_at_least_20=int(supported.sum())>=20,
       endpoint_normalized_median_at_most_3px=norm_stats['median'] is not None and norm_stats['median']<=3,
       cumulative_cycle_p95_effective_at_most_3px=cy_stats['p95'] is not None and cy_stats['p95']<=3)
  row=dict(path='→'.join(f'T{i}' for i in range(1,k+3)),reference='T1',
      automatic_endpoint_points=len(pa),automatic_evaluable_endpoint_points=int(supported.sum()),automatic_unmappable_endpoint_points=int((~supported).sum()),automatic_effective_endpoint_points=int(good.sum()),automatic_mean_error_effective_T1_px=stats(err_t1[good])['mean'],automatic_mean_error_T1_px=error_t1_stats['mean'],
      automatic_median_error_T1_px=error_t1_stats['median'],automatic_rmse_error_T1_px=error_t1_stats['rmse'],
      automatic_p95_error_T1_px=error_t1_stats['p95'],automatic_mean_error_native_period_px=error_period_stats['mean'],
      geometric_area_T1_px2=g,geometric_pct_T1=100*g/total,
      effective_area_T1_px2=e,effective_pct_T1=100*e/total,
      low_confidence_geometric_area_T1_px2=low,low_confidence_pct_T1=100*low/total,
      low_confidence_pct_geometric=100*low/g if g else None,
      other_unreliable_geometric_pct_T1=100*(g-e-low)/total,nonoverlap_pct_T1=100*(total-g)/total,
      cumulative_cycle_median_effective_T1_px=cy_stats['median'],cumulative_cycle_p95_effective_T1_px=cy_stats['p95'],
      automatic_qc_pass=all(checks.values()))
  rows.append(row);evaluations.append(dict(summary=row,automatic_checks=checks,
     independent_endpoint_matching=mi,endpoint_error_T1_all=error_t1_stats,
     endpoint_error_T1_effective=stats(err_t1[good]),endpoint_unmappable_count=int((~supported).sum()),endpoint_error_native_period=error_period_stats,
     endpoint_error_analysis_width1400=norm_stats))
  print(json.dumps(row,ensure_ascii=False),flush=True)
 save_native('T1_original_RGB.png',t1)
 common_geo=np.logical_and.reduce(geoms);common_valid=np.logical_and.reduce(valids)
 common_conf=np.minimum.reduce(confs);crop=rectangle(common_geo)
 root_mask_path=ROOT/'0353-1_root_priority_configuration'/POSITION/'T1_root_support_candidates.png'
 root_mask=(np.array(Image.open(root_mask_path))>0) if root_mask_path.exists() else None
 x0,y0,x1,y1=crop;sl=np.s_[y0:y1,x0:x1]
 save_png('four_period_common_geometric_mask_T1.png',common_geo.astype(np.uint8)*255)
 save_png('four_period_common_effective_mask_T1.png',common_valid.astype(np.uint8)*255)
 save_png('four_period_path_min_confidence_16bit_T1.png',np.rint(common_conf*65535).astype(np.uint16))
 crop_rgb=[a[sl] for a in aligned]
 for i,a in enumerate(crop_rgb,1):save_png(f'T{i}_registered_common_rectangle_RGB.png',a)
 paths=[r['path'] for r in rows]
 montage(crop_rgb,['T1 苗期（基准）','T2 抽穗期 → T1坐标','T3 成熟期 → T1坐标','T4 枯黄期 → T1坐标'],
         '01_registered_sequence_composite.png','统一尺度；原始RGB；相邻变换串联')
 # Common display scale for all three flow panels, in T1-size-normalized pixels.
 mags=[]
 for m,im,g in zip(maps,sources[1:],geoms):
  th,tw=im.shape[:2];disp=(m[sl]+.5)*np.array([w/tw,h/th],np.float32)-.5-base[sl]
  mags.append(np.linalg.norm(disp,axis=-1).ravel())
 scale=float(np.percentile(np.concatenate(mags),99))
 flows=[flow_preview(m,tuple(reversed(im.shape[:2])),base,g,scale)[sl] for m,im,g in zip(maps,sources[1:],geoms)]
 montage(flows,paths,'02_cumulative_transform_composite.png',f'方向HSV；亮度log1p；幅度上限{scale:.0f} T1像素；箭头仅方向')
 import matplotlib;matplotlib.use('Agg');from matplotlib import colormaps
 cm=colormaps['viridis']
 cmap=[np.rint(cm(c[sl])[...,:3]*255).astype(np.uint8) for c in confs]
 montage(cmap,paths,'03_cumulative_confidence_composite.png','置信度 0–1；路径上双向置信度最小值；紫低黄高')
 overlay=[np.rint((crop_rgb[0].astype(np.float32)+a)/2).astype(np.uint8) for a in crop_rgb[1:]]
 montage(overlay,paths,'04_overlay_50pct_composite.png','T1 与当前时期各50%；低置信区仅供质控')
 yy,xx=np.ogrid[y0:y1,x0:x1];tile=((xx//128+yy//128)%2)==0
 checkers=[]
 for a in crop_rgb[1:]:
  cb=a.copy();cb[tile]=crop_rgb[0][tile];checkers.append(cb)
 montage(checkers,paths,'05_checkerboard_composite.png','T1 / 当前时期交替；128个T1像素格')
 montage([np.repeat((v[sl]*255).astype(np.uint8)[...,None],3,axis=2) for v in valids],
         paths,'06_effective_overlap_mask_composite.png','白色严格有效；最后一格为四期共同有效区域')
 # Joint four-period quality partition on the T1 grid, without hiding failures.
 partition=np.zeros((h,w),np.uint8);partition[common_geo&~common_valid]=2
 partition[common_geo&(common_conf<THRESH)]=3;partition[common_valid]=1
 colors=np.array([[210,210,210],[48,154,96],[150,112,170],[232,162,47]],np.uint8)
 save_png('four_period_joint_quality_map_T1.png',colors[partition])
 summary=dict(position=POSITION,chain='T1→T2→T3→T4',reference_size_wh=[w,h],reference='native T1',
  error_coordinate='T1 original pixel coordinates for all endpoints',
  cumulative_confidence='minimum of all source- and target-side RoMa confidences sampled along adjacent forward path; not calibrated probability',
  effective_definition='each adjacent native target valid mask propagated to T1, full path in original image bounds, path confidence >=0.5, cumulative T1 round-trip <=3 px, positive cumulative Jacobian; nested across stages',
  common_geometric_area_T1_px2=int(common_geo.sum()),common_geometric_pct_T1=100*float(common_geo.mean()),
  common_effective_area_T1_px2=int(common_valid.sum()),common_effective_pct_T1=100*float(common_valid.mean()),
  common_low_confidence_geometric_area_T1_px2=int((common_geo&(common_conf<THRESH)).sum()),
  common_low_confidence_pct_T1=100*float((common_geo&(common_conf<THRESH)).mean()),
  largest_common_geometric_horizontal_rectangle_xyxy=crop,
  rectangle_size_wh=[x1-x0,y1-y0],rectangle_effective_fraction=float(common_valid[sl].mean()),
  rectangle_common_pixels=(x1-x0)*(y1-y0),full_chain_automatic_qc_pass=rows[-1]['automatic_qc_pass'],
  root_candidate_reference_area_T1_px2=int(root_mask.sum()) if root_mask is not None else None,
  root_candidate_common_effective_area_T1_px2=int((root_mask&common_valid).sum()) if root_mask is not None else None,
  root_candidate_common_effective_pct=100*float(common_valid[root_mask].mean()) if root_mask is not None and root_mask.any() else None,
  cumulative_paths_passed=sum(r['automatic_qc_pass'] for r in rows),cumulative_paths_evaluated=3,cumulative_paths_auto_qc_pct=100*sum(r['automatic_qc_pass'] for r in rows)/3,full_chains_evaluated=1,full_chains_automatic_qc_passed=int(rows[-1]['automatic_qc_pass']),full_chain_automatic_qc_pass_pct=100*int(rows[-1]['automatic_qc_pass']),
  manual_landmark_TRE=None,flow_display_max_T1_normalized_px=scale,flow_display_brightness='log1p magnitude / log1p shared p99 max; arrows normalized to fixed length indicate direction only',
  method_notes=['All images are sampled once from original RGB through composed RoMa fields.',
    'Endpoint checks never refit the dense field; coarse prealignment, if configured, can use related feature anchors.',
    'RGB display rectangle is geometric common coverage, not an all-high-confidence rectangle.',
    'Automatic quality thresholds: effective coverage>=10%, >=20 endpoint points, median error at width1400<=3px, effective roundtrip P95<=3 T1 pixels.',
    'Unmappable endpoint matches are counted separately and excluded from numeric error; no confidence/error-based deletion from the geometrically mappable headline error.',
    'Manual TRE and physically calibrated area unavailable. Automatic feature matching may be mistaken.',
    f'Only {POSITION} is evaluated in this report; success on 19 locations has not been evaluated.'],
  original_inference=json.loads((PAIR/'manifest.json').read_text()),cumulative_evaluations=evaluations)
 adj_report=ROOT/f'{POSITION}_registration_evaluation/evaluation_summary.json'
 if adj_report.exists():
  d=json.loads(adj_report.read_text());summary['adjacent_qc_passed_pairs']=d['automatic_qc_passed_pairs'];summary['adjacent_pairs']=d['evaluated_pairs']
 else:
  adjacent_summaries=[json.loads((PAIR/n/'metrics.json').read_text()) for n in ['T1_to_T2','T2_to_T3','T3_to_T4']]
  summary['adjacent_inference_completed_pairs']=3
  summary['adjacent_pairs']=3
  summary['adjacent_qc_passed_pairs']=sum(m['effective_area_ratio']>=.1 and m['reliable_sampled_matches']>=100 for m in adjacent_summaries)
  summary['adjacent_qc_definition']='Provisional internal check only: effective native target area >=10% and >=100 reliable RoMa samples; no human landmarks.'
  summary['adjacent_root_candidate_qc']=[{k:m.get(k) for k in ['pair','effective_area_ratio','root_candidate_target_area_ratio','root_candidate_effective_area_ratio','artifact_excluded_effective_pixels']} for m in adjacent_summaries]
 (OUT/'continuous_evaluation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 save_csv(OUT/'continuous_evaluation.csv',rows)
 np.savez_compressed(FIELDS/'four_period_common_masks_T1.npz',geometric_mask=common_geo,effective_mask=common_valid,
   confidence_path_min=common_conf.astype(np.float32),T1_size_wh=np.array([w,h]),crop_xyxy=np.array(crop))
 write_report(summary,rows)
 verify(summary,rows,aligned,maps,confs,valids,geoms)
 print('COMMON',json.dumps({k:v for k,v in summary.items() if k.startswith('common_') or k.startswith('rectangle') or k.startswith('full_chain')},ensure_ascii=False),flush=True)


def write_report(summary,rows):
 import os;os.environ.setdefault('MPLCONFIGDIR','/workspace/.cache/matplotlib')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib import font_manager
 from matplotlib.backends.backend_pdf import PdfPages
 font_manager.fontManager.addfont(REG)
 plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':42,'font.size':11})
 fig=plt.figure(figsize=(15,10),facecolor='white');fig.suptitle(f'{POSITION}  四期连续配准终点评估',fontsize=22,y=.97)
 adj=f"{summary.get('adjacent_qc_passed_pairs','未评估')}/{summary.get('adjacent_pairs',3)}"
 fig.text(.04,.91,f"相邻自动质控：{adj}   |   完整链自动质控：{'通过' if summary['full_chain_automatic_qc_pass'] else '未通过'}   |   人工标志点误差：未评估",fontsize=13)
 headers=['累计配准路径','可评估/自动点','平均误差\n统一T1像素','有效区平均误差\n统一T1像素','几何覆盖\n% T1','有效覆盖\n% T1','低置信面积\n% T1','自动质控']
 values=[]
 for r in rows:
  values.append([r['path'],f"{r['automatic_evaluable_endpoint_points']}/{r['automatic_endpoint_points']}",format(r['automatic_mean_error_T1_px'],'.3f') if r['automatic_mean_error_T1_px'] is not None else 'NA',
  format(r['automatic_mean_error_effective_T1_px'],'.3f') if r['automatic_mean_error_effective_T1_px'] is not None else 'NA',f"{r['geometric_pct_T1']:.2f}",f"{r['effective_pct_T1']:.2f}",f"{r['low_confidence_pct_T1']:.2f}",'通过' if r['automatic_qc_pass'] else '未通过'])
 ax=fig.add_axes([.025,.67,.95,.2]);ax.axis('off');table=ax.table(cellText=values,colLabels=headers,cellLoc='center',bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(10)
 for (i,j),cell in table.get_celld().items():
  cell.set_facecolor('#e9f0f5' if i==0 else ('#f7f9fb' if i%2==0 else 'white'));cell.set_edgecolor('#d6dfe5')
 ax=fig.add_axes([.065,.34,.43,.25]);bottom=np.zeros(3);x=np.arange(3)
 for key,label,c in [('effective_pct_T1','有效','#309a60'),('low_confidence_pct_T1','低置信','#e8a22f'),('other_unreliable_geometric_pct_T1','其他不可靠','#9670aa'),('nonoverlap_pct_T1','无共同几何来源','#d2d2d2')]:
  a=np.array([r[key] for r in rows]);ax.bar(x,a,bottom=bottom,width=.58,label=label,color=c);bottom+=a
 ax.set_ylim(0,100);ax.set_xticks(x,['至 T2','至 T3','至 T4']);ax.set_ylabel('T1 完整画布面积 (%)');ax.set_title('可靠区域沿路径逐步收缩');ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)
 ax2=fig.add_axes([.59,.34,.34,.25]);ax2.bar(x,[r['automatic_mean_error_T1_px'] if r['automatic_mean_error_T1_px'] is not None else np.nan for r in rows],color='#3978a9',width=.58);ax2.set_xticks(x,['至 T2','至 T3','至 T4']);ax2.set_ylabel('平均误差（T1 原始像素）');ax2.set_title('自动终点匹配检查累计误差');ax2.spines[['top','right']].set_visible(False)
 notes=[f"四期共同严格有效面积：{summary['common_effective_area_T1_px2']:,} px²（T1 全画布的 {summary['common_effective_pct_T1']:.2f}%）。",
 f"四期共同几何覆盖：{summary['common_geometric_pct_T1']:.2f}%；路径低置信度面积：{summary['common_low_confidence_pct_T1']:.2f}%（分母同为T1全画布）。",
 f"展示裁剪为最大共同几何横向矩形 {summary['rectangle_size_wh'][0]}×{summary['rectangle_size_wh'][1]}；其中有效像素占 {100*summary['rectangle_effective_fraction']:.2f}%。",
 '累计变换由 T1→T2、T2→T3、T3→T4 串联得到；所有指标与显示均使用 T1 原始尺度。',
 '置信度是路径上各段双向置信度的最小值。有效性同时传播各段掩膜，并检查累计往返误差≤3个T1像素。',
 'RootSIFT自动终点复核；粗配准可用相关特征点，故非独立人工真值。无路径点单独计数；有效区误差另列。',
 '自动通过阈值：有效面积≥10%、终点点数≥20、统一分析尺度中位误差≤3px、有效区累计循环P95≤3个T1像素。',
 '共同几何覆盖不等于可靠重叠；低置信及掩膜外的形变不能作为定量根系追踪证据。']
 for j,n in enumerate(notes):fig.text(.04,.235-j*.024,n,fontsize=9,color='#38424b')
 fig.savefig(OUT/'continuous_evaluation_summary.png',dpi=240,bbox_inches='tight')
 fig2,axes=plt.subplots(6,1,figsize=(16,12),facecolor='white')
 for ax,n,label in zip(axes,['01_registered_sequence_composite.png','02_cumulative_transform_composite.png','03_cumulative_confidence_composite.png','04_overlay_50pct_composite.png','05_checkerboard_composite.png','06_effective_overlap_mask_composite.png'],['配准图','累计变换场','累计置信度','50% 叠加','棋盘格质控','有效重叠掩膜']):
  im=Image.open(OUT/n);im.thumbnail((2300,700));ax.imshow(im);ax.axis('off');ax.text(-.015,.5,label,ha='right',va='center',transform=ax.transAxes,fontsize=11)
 fig2.suptitle('T1 坐标系：四期连续配准综合结果',fontsize=18,y=.99);fig2.tight_layout(rect=[.05,0,1,.98])
 fig2.savefig(OUT/'continuous_registration_overview.png',dpi=200)
 with PdfPages(OUT/'continuous_registration_evaluation.pdf') as pdf:pdf.savefig(fig);pdf.savefig(fig2)
 plt.close('all')


def verify(summary,rows,aligned,maps,confs,valids,geoms):
 h,w=aligned[0].shape[:2]
 for k,(m,c,v,g) in enumerate(zip(maps,confs,valids,geoms)):
  z=np.load(FIELDS/f'T1_to_T{k+2}_cumulative_on_T1.npz');np.testing.assert_array_equal(z['effective_mask_T1'],v);np.testing.assert_array_equal(z['T1_grid_to_period_xy_px'],m);z.close()
  assert np.isfinite(m).all() and np.isfinite(c).all() and c.min()>=0 and c.max()<=1
  assert not (v&~g).any() and not (v&(c<THRESH)).any()
  if k:assert not (v&~valids[k-1]).any() and not (g&~geoms[k-1]).any()
  assert Image.open(DETAIL/f'T{k+2}_registered_on_T1_RGB.png').size==(w,h)
  assert rows[k]['effective_area_T1_px2']==int(v.sum())
  total=sum(rows[k][n] for n in ['effective_pct_T1','low_confidence_pct_T1','other_unreliable_geometric_pct_T1','nonoverlap_pct_T1']);assert abs(total-100)<1e-8
 x0,y0,x1,y1=summary['largest_common_geometric_horizontal_rectangle_xyxy']
 assert np.logical_and.reduce(geoms)[y0:y1,x0:x1].all()
 assert np.array_equal(np.array(Image.open(DETAIL/'T1_original_RGB.png')),aligned[0])
 for i in range(1,5):assert Image.open(OUT/f'T{i}_registered_common_rectangle_RGB.png').size==(x1-x0,y1-y0)
 (OUT/'validation.json').write_text(json.dumps(dict(status='PASS',checks=['native T1 canvas sizes','dense-map lossless save/reload','confidence bounds','nested propagated geometric/effective masks','area partition sums','all-period geometric rectangle','unchanged T1 RGB','four equally sized ROI RGBs']),indent=2))
 print('PASS: continuous field, masks, dimensions, area partition, and RGB checks',flush=True)

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--position',default='0353-2-B3')
 args=parser.parse_args();configure_position(args.position);main()
