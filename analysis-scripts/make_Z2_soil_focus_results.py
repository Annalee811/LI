"""Export explicit soil-structure coarse alignment, distinct from accepted RoMa dense registration."""
from pathlib import Path
import json,sys,cv2,numpy as np,h5py
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_continuous_registration as v
ROOT=Path.cwd()/'results/microtube_registration';RAW=ROOT/'input/raw/V-变化/A017-1-Z2'
OUT=ROOT/'A017-1-Z2_soil_structure_coarse_results';OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4)
rows=json.loads((ROOT/'A017-1_Z2_soil_shapes/soil_shape_candidates.json').read_text())
h14=np.array(next(d['matrix_native'] for d in rows if d['mode']=='clahe' and d['pair']=='T1_to_T4'))
h24=np.array(next(d['matrix_native'] for d in rows if d['mode']=='clahe' and d['pair']=='T2_to_T4'))
matrices={2:np.linalg.inv(h24)@h14,4:h14}
images={t:np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB')) for t in [1,2,4]}
h,w=images[1].shape[:2];base=v.grid(h,w);mappings={t:cv2.perspectiveTransform(base,m.astype(np.float64)).astype(np.float32) for t,m in matrices.items()}
geoms={t:v.inside(m,Image.open(RAW/f'T{t}.jpg').size) for t,m in mappings.items()}
common=np.logical_and.reduce(list(geoms.values()));crop=v.rectangle(common);x0,y0,x1,y1=crop;sl=np.s_[y0:y1,x0:x1]
aligned={1:images[1]}
for t,m in mappings.items():
 a=cv2.remap(images[t],m[...,0],m[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);a[~geoms[t]]=0;aligned[t]=a
for t in [1,2,4]:Image.fromarray(aligned[t][sl]).save(OUT/f'T{t}_soil_structure_coarse_RGB.png',dpi=(600,600))
Image.fromarray(common.astype(np.uint8)*255).save(OUT/'soil_structure_geometric_common_mask_T1.png')
oldout=v.OUT;v.OUT=OUT
v.montage([aligned[t][sl] for t in [1,2,4]],['T1 基准','T2 土壤结构粗对齐','T4 土壤结构粗对齐'],'soil_structure_coarse_RGB_montage.png','T3缺失；原始RGB；结构粗对齐，不能代替RoMa有效性验证')
v.montage([np.rint((aligned[1][sl].astype(np.float32)+aligned[t][sl])/2).astype(np.uint8) for t in [2,4]],['T1 / T2','T1 / T4'],'soil_structure_coarse_overlay_50pct.png','结构粗对齐，各时期50%；不是已通过质控的稠密配准')
yy,xx=np.ogrid[y0:y1,x0:x1];tile=(yy//96+xx//96)%2==0;check=[]
for t in [2,4]:
 a=aligned[t][sl].copy();a[tile]=aligned[1][sl][tile];check.append(a)
v.montage(check,['T1 / T2','T1 / T4'],'soil_structure_coarse_checkerboard.png','96个T1像素格；结构粗对齐诊断')
v.OUT=oldout
# Observation windows correspond to soil crack outlines shown by the user, not landmark truth.
centers_analysis=np.array([[375,1035],[610,1075],[715,1050]],np.float32)
centers=centers_analysis*np.array([w/1400,h/round(h*1400/w)],np.float32)
roi_rgb=[Image.fromarray(aligned[t][sl]).convert('RGB') for t in [1,2,4]]
panelw=x1-x0;panelh=y1-y0;header=135;footer=85;canvas=Image.new('RGB',(3*panelw,panelh+header+footer),'white');draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype(v.FONT,34);small=ImageFont.truetype(v.REG,24)
windows=[]
for j,t in enumerate([1,2,4]):
 canvas.paste(roi_rgb[j],(j*panelw,header));draw.text((j*panelw+20,15),f'T{t}：土壤裂缝结构观察',font=font,fill=(30,36,42))
 draw.text((j*panelw+20,70),'T1坐标；T3缺失；原色RGB；仅为结构粗对齐',font=small,fill=(65,75,85))
 for label,(cx,cy) in zip(['A','B','C'],centers):
  halfx,halfy=95,65;bx0=max(0,int(cx-x0-halfx));by0=max(0,int(cy-y0-halfy));bx1=min(panelw-1,int(cx-x0+halfx));by1=min(panelh-1,int(cy-y0+halfy))
  if bx0>=bx1 or by0>=by1:continue
  draw.rectangle((j*panelw+bx0,header+by0,j*panelw+bx1,header+by1),outline=(28,116,155),width=4)
  draw.text((j*panelw+bx0+4,header+by0+2),label,font=font,fill=(20,105,150))
  if j==0:windows.append(dict(label=label,T1_center_xy=[float(cx),float(cy)],T1_window_xyxy=[x0+bx0,y0+by0,x0+bx1,y0+by1]))
 draw.text((j*panelw+20,header+panelh+15),'A–C 为结构观察窗，非人工标志点；可靠性以RoMa掩膜为准',font=small,fill=(65,75,85))
canvas.save(OUT/'soil_structure_focus_figure.png',dpi=(600,600),compress_level=5)
comparison=Image.new('RGB',(1660,1140),'white');cd=ImageDraw.Draw(comparison);cf=ImageFont.truetype(v.REG,24)
for k,t in enumerate([1,2,4]):cd.text((140+k*500,10),f'T{t} 原色RGB（结构粗对齐）',font=cf,fill=(30,40,50))
for j,window in enumerate(windows):
 cd.text((20,90+j*340),window['label']+' 结构窗',font=cf,fill=(25,100,150));bx0,by0,bx1,by1=window['T1_window_xyxy']
 for k,t in enumerate([1,2,4]):
  patch=Image.fromarray(aligned[t][sl]).crop((bx0-x0,by0-y0,bx1-x0+1,by1-y0+1));patch.save(OUT/f'T{t}_soil_window_{window["label"]}_RGB.png',dpi=(600,600));patch.thumbnail((490,300));patch=patch.resize((490,round(490*patch.height/patch.width)),Image.Resampling.LANCZOS);comparison.paste(patch,(140+k*500,60+j*340))
cd.text((20,1080),'A–C为观察窗口，非人工标志点；仅放大显示；T3缺失；不代表连续RoMa配准已通过质控',font=cf,fill=(55,65,75));comparison.save(OUT/'soil_structure_window_comparison.png',dpi=(600,600))
with h5py.File(OUT/'soil_structure_affine_fields_ROI.h5','w') as f:
 f.attrs['field_type']='COARSE AFFINE SOIL LOCALIZATION, not a RoMa dense field; no confidence assigned'
 f.attrs['reference']='original T1 pixel centres (x,y)';f.attrs['T1_size_wh']=[w,h];f.attrs['T1_crop_xyxy']=crop;f.attrs['period_ids']=[1,2,4]
 for t,m in mappings.items():
  g=f.create_group(f'T{t}');g.create_dataset('T1_crop_grid_to_period_xy_px',data=m[sl],compression='gzip',compression_opts=6,shuffle=True,fletcher32=True)
  g.create_dataset('T1_to_period_affine',data=matrices[t]);g.attrs['target_size_wh']=Image.open(RAW/f'T{t}.jpg').size
notes=dict(position='A017-1-Z2',period_ids=[1,2,4],missing_period_ids=[3],field_type='coarse affine soil-structure localization; NOT accepted continuous RoMa registration',manual_landmark_TRE=None,
 coarse_method='T1/T4 and T2/T4 stable soil descriptors; T4 bridge gives T1/T2 prior. All three RGBs sampled once from originals onto T1.',
 original_T1_size_wh=[w,h],common_geometric_crop_xyxy=crop,observation_windows=windows,affine_matrices={f'T1_to_T{t}':m.tolist() for t,m in matrices.items()},
 geometric_common_pct_T1=100*float(common.mean()),automatic_fit_summaries=[{k:d[k] for k in ['mode','pair','mutual_matches','inliers','median_fit_error_analysis_px'] if k in d} for d in rows],
 interpretation='The marked shapes support coarse localization. Fitting residuals are not independent registration errors or manual TRE. Low-confidence RoMa regions cannot support quantitative root tracing.')
(OUT/'soil_structure_coarse_summary.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2))
for t in [1,2,4]:
 np.testing.assert_array_equal(np.array(Image.open(OUT/f'T{t}_soil_structure_coarse_RGB.png')),aligned[t][sl])
assert common[sl].all();print('PASS soil coarse RGB and geometric crop',crop,flush=True)
