#!/usr/bin/env python3
"""Usable T2→T3→T4 supplement, explicitly separate from failed T1-based chain."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,h5py
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_continuous_registration as v
ROOT=Path.cwd()/'results/microtube_registration';POS='0353-1-Z2'
v.configure_position(POS);out=ROOT/f'{POS}_T2_three_period_supplement';out.mkdir(exist_ok=True);v.OUT=out
raw=ROOT/'input/raw/V-变化'/POS
images=[v.rgb(next(f for f in raw.iterdir() if f.stem==f'T{i}')) for i in [2,3,4]]
h,w=images[0].shape[:2];base=v.grid(h,w);f23=v.load('T2_to_T3');f34=v.load('T3_to_T4');fields=[f23,f34]
map_now=base.copy();geo=np.ones((h,w),bool);valid=geo.copy();conf=np.ones((h,w),np.float32)
aligned=[images[0]];maps=[];confs=[];geos=[];valids=[];cycles=[]
for k,f in enumerate(fields):
 old=map_now.copy();map_now=v.remap(f['source_to_target_xy_px'],old)
 geo=geo&v.inside(old,tuple(f['source_size_wh']))&v.inside(map_now,tuple(f['target_size_wh']))
 conf=np.minimum(conf,np.minimum(v.remap(f['confidence_source'],old,border=0),v.remap(f['confidence_target'],map_now,border=0)));conf[~geo]=0
 back=map_now.copy()
 for z in fields[:k+1][::-1]:back=v.remap(z['target_to_source_xy_px'],back)
 cycle=np.linalg.norm(back-base,axis=-1)
 valid=valid&geo&(v.remap(f['valid_target'].astype(np.uint8),map_now,True,border=0)>0)&(conf>=.5)&(cycle<=3)&(v.jac(map_now)>0)
 maps.append(map_now.copy());confs.append(conf.copy());geos.append(geo.copy());valids.append(valid.copy());cycles.append(cycle.copy())
 a=v.remap(images[k+1],map_now,border=0);a[~geo]=0;aligned.append(a)
common_geo=np.logical_and.reduce(geos);common_valid=np.logical_and.reduce(valids);crop=v.rectangle(common_geo);x0,y0,x1,y1=crop;sl=np.s_[y0:y1,x0:x1]
for i,a in zip([2,3,4],aligned):
 v.save_png(f'T{i}_registered_on_T2_RGB.png',a);v.save_png(f'T{i}_common_rectangle_RGB.png',a[sl])
paths=['T2→T3','T2→T3→T4']
v.montage([a[sl] for a in aligned],['T2 基准','T3→T2坐标','T4→T2坐标'],'01_registered_sequence_composite.png','补充三期结果；不包含未可靠配准的T1')
mags=[]
for m,a in zip(maps,images[1:]):
 ih,iw=a.shape[:2];d=(m[sl]+.5)*np.array([w/iw,h/ih])-.5-base[sl];mags.append(np.linalg.norm(d,axis=-1).ravel())
scale=float(np.percentile(np.concatenate(mags),99))
v.montage([v.flow_preview(m,tuple(reversed(a.shape[:2])),base,g,scale)[sl] for m,a,g in zip(maps,images[1:],geos)],paths,'02_cumulative_transform_composite.png','补充三期累计场；箭头表示方向')
from matplotlib import colormaps
cmap=colormaps['viridis'];v.montage([np.rint(cmap(c[sl])[...,:3]*255).astype(np.uint8) for c in confs],paths,'03_cumulative_confidence_composite.png','路径双向置信度最小值；0–1')
rs=[a[sl] for a in aligned];v.montage([np.rint((rs[0].astype(np.float32)+a)/2).astype(np.uint8) for a in rs[1:]],paths,'04_overlay_50pct_composite.png','T2 与后一期各50%')
yy,xx=np.ogrid[y0:y1,x0:x1];choose=(xx//64+yy//64)%2==0;cbs=[]
for a in rs[1:]:
 cb=a.copy();cb[choose]=rs[0][choose];cbs.append(cb)
v.montage(cbs,paths,'05_checkerboard_composite.png','64个T2像素格；仅质控')
v.montage([np.repeat((m[sl]*255).astype(np.uint8)[...,None],3,axis=2) for m in valids],paths,'06_effective_overlap_mask_composite.png','白色严格有效；仅T2、T3、T4共同区域')
v.save_png('three_period_common_effective_mask_T2.png',common_valid.astype(np.uint8)*255);v.save_png('three_period_common_geometric_mask_T2.png',common_geo.astype(np.uint8)*255)
v.save_png('three_period_path_confidence_16bit_T2.png',np.rint(confs[-1]*65535).astype(np.uint16))
with h5py.File(out/'T2-T4_three_period_dense_fields.h5','w') as hf:
 hf.attrs['reference']='native T2 pixel coordinates; supplement excludes T1'
 hf.attrs['chain']='T2 -> T3 -> T4';hf.attrs['reference_size_wh']=[w,h];hf.attrs['crop_xyxy']=crop
 for k,t in enumerate([3,4]):
  group=hf.create_group(f'T{t}')
  for name,array in [('T2_grid_to_period_xy_px',maps[k]),('confidence_path_min',confs[k]),('effective_mask_T2',valids[k]),('geometric_mask_T2',geos[k])]:group.create_dataset(name,data=array,compression='gzip',compression_opts=6,shuffle=True,fletcher32=True)
  group.attrs['period_size_wh']=list(reversed(images[k+1].shape[:2]))
 summary=dict(position=POS,reference='T2',chain='T2→T3→T4',supplement_excludes_T1=True,
  common_effective_pixels=int(common_valid.sum()),common_effective_pct_T2=100*float(common_valid.mean()),
  common_geometric_pct_T2=100*float(common_geo.mean()),crop_xyxy=crop,crop_size_wh=[x1-x0,y1-y0],
  effective_pct_inside_crop=100*float(common_valid[sl].mean()),note='This three-period result does not establish four-period continuity from T1.')
(out/'three_period_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(v.REG);plt.rcParams['font.family']='Noto Sans CJK JP'
fig,axes=plt.subplots(6,1,figsize=(14,10))
for ax,name,label in zip(axes,sorted(out.glob('0[1-6]_*.png')),['配准图','变换场','置信度','50%叠加','棋盘格','有效掩膜']):
 im=Image.open(name);im.thumbnail((1800,600));ax.imshow(im);ax.axis('off');ax.text(-.015,.5,label,ha='right',va='center',transform=ax.transAxes)
fig.suptitle('0353-1-Z2｜T2→T3→T4 补充连续结果（不含T1）',fontsize=17);fig.tight_layout(rect=[.06,0,1,.96]);fig.savefig(out/'three_period_results_overview.png',dpi=220);plt.close(fig)
assert all(Image.open(out/f'T{i}_common_rectangle_RGB.png').size==(x1-x0,y1-y0) for i in [2,3,4]);assert common_geo[sl].all()
with h5py.File(out/'T2-T4_three_period_dense_fields.h5','r') as hf:
 for k,t in enumerate([3,4]):np.testing.assert_array_equal(hf[f'T{t}/T2_grid_to_period_xy_px'][...],maps[k])
print('PASS: supplemental original RGB, shared crop, masks, and lossless fields',json.dumps(summary,ensure_ascii=False),flush=True)
