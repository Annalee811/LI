#!/usr/bin/env python3
"""Preserve honest Q1 partial outcomes: no invented common area for unlocalized images.
Q1-2-B3: model-only T1/T2/T3 proposal, T4 unlocated.
Q1-3-Z3: no T1-referenced location; only original RGB and review figure.
"""
from pathlib import Path
import json,hashlib,zipfile,argparse
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import B2_region_tools as u
p=argparse.ArgumentParser();p.add_argument('--position',required=True,choices=['Q1-2-B3','Q1-3-Z3']);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);args=p.parse_args();root=args.root.resolve();pos=args.position;raw=root/'input/raw/V-变化'/pos;out=root/(pos+'_region_localization');images=out/'images';images.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(2);cv2.setRNGSeed(20261013);files={int(f.stem[1:]):f for f in raw.glob('T*')};ts=sorted(files);ims={t:np.array(Image.open(f).convert('RGB')) for t,f in files.items()};wh={t:ims[t].shape[1::-1] for t in ts};hashes={f'T{t}':hashlib.sha256(f.read_bytes()).hexdigest() for t,f in files.items()};manifest=json.loads((root/pos/'manifest.json').read_text());expected={Path(k).stem:(v['sha256'] if isinstance(v,dict) else v) for k,v in manifest['input_sha256'].items()};assert hashes==expected
selected=[];native={};box=None;polygons={};evidence=[];image_names=[]
if pos=='Q1-2-B3':
 # These are explicit neural hypotheses with no sufficient independent descriptor support.
 # They are not called accepted full-period registration or observed root identities.
 localized=[1,2,3];unlocalized=[4];native={1:np.eye(3)};W=1600
 for a,b in [(1,2),(2,3)]:
  name=f'T{a}_to_T{b}';z=np.load(root/pos/name/'sampled_matches.npz');ok=z['reliable'];pa=z['source_xy'][ok]*W/wh[a][0];pb=z['target_xy'][ok]*W/wh[b][0];assert len(pa)>=100;H,k=cv2.estimateAffinePartial2D(pa.astype('float32'),pb.astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999);assert H is not None and k.sum()>=100;H=np.vstack([H,[0,0,1]]);scale=np.hypot(H[0,0],H[1,0]);angle=np.degrees(np.arctan2(H[1,0],H[0,0]));assert .8<scale<1.2 and abs(angle)<10;pair_native=np.diag([wh[b][0]/W]*2+[1])@H@np.diag([W/wh[a][0]]*2+[1]);native[b]=pair_native@native[a];selected.append(name);evidence.append(dict(pair=name,internally_supported_candidates=int(ok.sum()),global_similarity_consensus=int(k.sum()),matrix_analysis1600=H.tolist(),validation='Neural internal support only; not independent accuracy or accepted region registration'))
 g=u.grid(ims[1].shape[0],ims[1].shape[1]);maps={t:u.project(g,native[t]) for t in localized};common=np.logical_and.reduce([u.inside(maps[t],wh[t]) for t in localized]);box=u.rectangle(common);x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1]
 for t in localized:
  crop=u.remap(ims[t],maps[t][sl]);n=f'T{t}_candidate_T1_region_RGB.png';Image.fromarray(crop).save(images/n,dpi=(600,600));np.testing.assert_array_equal(np.array(Image.open(images/n)),u.remap(ims[t],maps[t][sl]));s=np.linalg.svd(native[t][:2,:2],compute_uv=False);np.testing.assert_allclose(s[0],s[1],rtol=1e-10);image_names.append(n);polygons[f'T{t}']=u.project(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),native[t]).tolist()
 np.testing.assert_array_equal(np.array(Image.open(images/'T1_candidate_T1_region_RGB.png')),ims[1][sl]);assert common[sl].all();n='T4_unlocalized_original_RGB.png';Image.fromarray(ims[4]).save(images/n,dpi=(600,600));np.testing.assert_array_equal(np.array(Image.open(images/n)),ims[4]);image_names.append(n);status='partial_neural_region_candidate';title='前三期仅为RoMa模型候选；T4尚未定位';dim=[x1-x0,y1-y0]
else:
 localized=[];unlocalized=[1];status='no_T1_referenced_localization';title='T1未可靠定位：仅原图核对，不作共同区域裁剪';dim=None
 for t in ts:
  n=f'T{t}_unregistered_original_RGB.png';Image.fromarray(ims[t]).save(images/n,dpi=(600,600));np.testing.assert_array_equal(np.array(Image.open(images/n)),ims[t]);image_names.append(n)
panelw=1600;header=180;footer=105;panelh=max(round(ims[t].shape[0]*panelw/ims[t].shape[1]) for t in ts);canvas=Image.new('RGB',(panelw*len(ts),panelh+header+footer),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype(u.FONT,35);small=ImageFont.truetype(u.FONT,24);phases={1:'苗期',2:'抽穗期',3:'成熟期',4:'枯黄期'}
for i,t in enumerate(ts):
 im=Image.fromarray(ims[t]);im=im.resize((panelw,round(im.height*panelw/im.width)),Image.Resampling.LANCZOS);canvas.paste(im,(i*panelw,header));lab='模型候选区域' if t in localized else ('T4未定位，仅原图核对' if pos=='Q1-2-B3' else '原图核对，未建立T1共同区域');d.text((i*panelw+16,12),f'T{t} {phases[t]}：{lab}',font=font,fill='#24343c');d.text((i*panelw+16,75),title,font=small,fill='#52626d');d.text((i*panelw+16,118),'保持原色；未经人工标志点验证',font=small,fill='#52626d')
 if t in localized:
  pts=np.array(polygons[f'T{t}'])*panelw/wh[t][0];pts=[(float(q[0]+i*panelw),float(q[1]+header)) for q in pts];d.line(pts+[pts[0]],fill='#9bd4df',width=2);caption=f'候选裁剪：{dim[0]}×{dim[1]} T1像素；非四期共同区'
 else:caption='独立保存原始RGB像素；不猜测裁剪坐标'
 d.text((i*panelw+16,header+panelh+12),caption,font=small,fill='#52626d');d.text((i*panelw+16,header+panelh+53),'该图不能证明完整连续配准成功',font=small,fill='#52626d')
name=pos+'_localization_review.png';canvas.save(images/name,dpi=(600,600));canvas.resize((2200,round(canvas.height*2200/canvas.width)),Image.Resampling.LANCZOS).save(images/(pos+'_review_preview.jpg'),quality=96,subsampling=0);image_names.insert(0,name)
meta=dict(position=pos,status=status,period_ids=ts,missing_period_ids=sorted(set(range(1,5))-set(ts)),localized_candidate_period_ids=localized,unlocalized_period_ids=unlocalized,full_T1_referenced_sequence_accepted=False,full_common_region_defined=False,candidate_crop_size_wh=dim,candidate_common_T1_rectangle_xyxy=box,matrices_T1_to_native_candidates={f'T{t}':H.tolist() for t,H in native.items()},original_canvas_candidate_polygons=polygons,neural_hypothesis_evidence=evidence,candidate_region_manual_review=True,manual_TRE=None,output_image_filenames=image_names,input_sha256=hashes,interpretation=title+'. Failure remains failure. Missing periods are not synthesized; unlocalized existing periods are not relabeled missing. Only existing supported neural hypotheses are rendered as explicitly unverified candidates. Original unsupported RGB remains native, uncropped and unregistered. No elastic deformation or color changes.',limitations='Confidence/cycle/model consensus are internal evidence, not validated root identity or ground truth. Full-frame descriptor checks, contrast normalization and orientation trials failed to locate the unlocalized image(s).')
(out/'localization_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));archive=root.parent/(pos+'_partial_localization_review_and_RGB.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for n in image_names:z.write(images/n,n)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None;assert set(z.namelist())==set(image_names)
assert archive.stat().st_size<100*1024**2;print('SAVED_PARTIAL_REVIEW',pos,status,dim,'unlocalized',unlocalized,'ZIP',archive.stat().st_size,flush=True)
