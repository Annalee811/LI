#!/usr/bin/env python3
"""Four-frame review with only explicitly supported partial T1 observation candidates."""
from pathlib import Path
import json, hashlib, zipfile, os
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
import B2_region_tools as u

r=Path(__file__).resolve().parent; pos='Q1-2-B3'; w=r/(pos+'_review_work')
out=w/'reviewed_results'; out.mkdir(exist_ok=True)
raw=r/'input/raw/V-变化'/pos
files={int(f.stem[1:]):f for f in raw.glob('T*')}; ims={t:np.array(Image.open(f).convert('RGB')) for t,f in files.items()}
meta=json.loads((w/'partial_123/band_A'/(pos+'_region_localization')/'localization_metadata.json').read_text())
assert meta['period_ids']==[1,2,3] and meta['candidate_region_manual_review']
native={int(t[1:]):np.array(H) for t,H in meta['matrices_T1_to_native'].items()}
box=meta['common_T1_rectangle_xyxy']; x0,y0,x1,y1=box
grid=u.grid(y1-y0,x1-x0)+[x0,y0]; crops={}; polygons={}; names=[]
for t in [1,2,3]:
 assert hashlib.sha256(files[t].read_bytes()).hexdigest()==meta['input_sha256'][f'T{t}']
 coords=u.project(grid,native[t]); assert u.inside(coords,ims[t].shape[1::-1]).all()
 s=np.linalg.svd(native[t][:2,:2],compute_uv=False); np.testing.assert_allclose(s[0],s[1],rtol=1e-10)
 crops[t]=u.remap(ims[t],coords); name=f'T{t}_candidate_T1_region_RGB.png'; names.append(name)
 Image.fromarray(crops[t]).save(out/name,dpi=(600,600)); np.testing.assert_array_equal(np.array(Image.open(out/name)),crops[t])
 polygons[t]=u.project(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),native[t])
np.testing.assert_array_equal(crops[1],ims[1][y0:y1,x0:x1])
name='T4_unlocalized_original_RGB.png'; target=out/name
source=Path('/workspace/LI-registration-upload/roma-results/Q1-2/Q1-2-B3/region-observation-results')/name
if not target.exists():os.link(source,target)
np.testing.assert_array_equal(np.array(Image.open(target)),ims[4]);names.append(name)
font=lambda n:ImageFont.truetype(u.FONT,n); phases={1:'苗期',2:'抽穗期',3:'成熟期',4:'枯黄期'}
pw=1600; ph=max(round(a.shape[0]*pw/a.shape[1]) for a in ims.values()); head=180
canvas=Image.new('RGB',(pw*4,ph+head+105),'white'); d=ImageDraw.Draw(canvas)
for t in [1,2,3,4]:
 ox=(t-1)*pw; im=Image.fromarray(ims[t]); im=im.resize((pw,round(im.height*pw/im.width)),Image.Resampling.LANCZOS); canvas.paste(im,(ox,head))
 d.text((ox+18,12),f'T{t} {phases[t]}：'+('前三期区域候选' if t<4 else '未定位，仅原图核对'),font=font(35),fill='#24343c')
 d.text((ox+18,76),'T1/T2/T3候选；没有四期共同观察区域',font=font(25),fill='#52626d')
 d.text((ox+18,122),'原色；无人工标志点；候选仍需复核',font=font(25),fill='#52626d')
 if t<4:
  pts=polygons[t]*pw/ims[t].shape[1]+[ox,head]; pts=[tuple(p) for p in pts];d.line(pts+[pts[0]],fill='#9bd4df',width=2)
 caption=f'独立候选裁剪：{x1-x0}×{y1-y0}' if t<4 else '保留原始RGB像素与尺寸；不猜测裁剪坐标'
 d.text((ox+18,head+ph+14),caption,font=font(25),fill='#52626d');d.text((ox+18,head+ph+60),'该图不证明完整四期连续配准成功',font=font(24),fill='#52626d')
name=pos+'_localization_review.png';canvas.save(out/name,dpi=(600,600));names.insert(0,name)
canvas.resize((2600,round(canvas.height*2600/canvas.width)),Image.Resampling.LANCZOS).save(out/(pos+'_review_preview.jpg'),quality=97,subsampling=0)
pw=1000;rh=round((y1-y0)*pw/(x1-x0));seq=Image.new('RGB',(3000,rh+180),'white');d=ImageDraw.Draw(seq)
for t in [1,2,3]:
 ox=(t-1)*pw;seq.paste(Image.fromarray(crops[t]).resize((pw,rh),Image.Resampling.LANCZOS),(ox,110));d.text((ox+15,12),f'T{t} {phases[t]} | 区域候选',font=font(32),fill='#24343c');d.text((ox+15,62),'仅前三期；T4存在但未定位',font=font(25),fill='#52626d');d.text((ox+15,rh+125),'保留局部变化；不判定根系身份',font=font(24),fill='#52626d')
name=pos+'_partial_candidate_sequence.png';seq.save(out/name,dpi=(600,600));names.append(name)
mask=np.zeros(ims[1].shape[:2],np.uint8);mask[y0:y1,x0:x1]=128;name='T1_partial_candidate_observation_mask.png';Image.fromarray(mask).save(out/name);names.append(name)
archive=w/(pos+'_partial_localization_review_and_RGB.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name in names:z.write(out/name,name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==7
assert archive.stat().st_size<100*1024**2
audit=dict(position=pos,source_period_ids=[1,2,3,4],localized_candidate_period_ids=[1,2,3],unlocalized_period_ids=[4],missing_period_ids=[],full_common_region_defined=False,full_T1_referenced_sequence_accepted=False,candidate_T1_rectangle_xyxy=box,candidate_RGB_wh=[x1-x0,y1-y0],subset_metadata=meta,source_sha256={f'T{t}':hashlib.sha256(f.read_bytes()).hexdigest() for t,f in files.items()},validation=dict(source_hashes_checked=True,T1_crops_pixel_exact=True,uniform_global_similarities_only=True,source_bounds_checked=True,T4_original_pixels_exact=True,zip_CRC_checked=True),limitations='Partial conditioned structure graph only; no qualifying extra route, direct T2/T3 insufficient anchors. T4 exists but remains unlocated. Partial mask128 is hypothesis coverage, not a full-period or confidence mask. No human TRE, root identity, elastic warp or colour change.')
(w/'partial_review_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print('PARTIAL_RENDER_VALIDATED',archive.stat().st_size,box,audit['candidate_RGB_wh'],flush=True)
