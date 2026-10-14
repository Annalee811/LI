#!/usr/bin/env python3
"""Publish original-colour observation regions with explicit non-data seams.

Region fits, conditional validation and all disagreements are preserved in a
separate audit. RGB is sampled once from originals through global similarities.
"""
from pathlib import Path
import argparse,json,hashlib,zipfile
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
import B2_region_tools as u

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--position',required=True);p.add_argument('--bands',nargs='+',required=True);p.add_argument('--windows',type=Path);p.add_argument('--threads',type=int,default=1);args=p.parse_args()
root=args.root.resolve();pos=args.position;cv2.setNumThreads(args.threads);out=root/'reviewed_results';out.mkdir(exist_ok=True)
raw=root/'input/raw/V-变化'/pos;paths={int(f.stem[1:]):f for f in raw.glob('T*')};ims={t:np.array(Image.open(f).convert('RGB')) for t,f in paths.items()};wh={t:a.shape[1::-1] for t,a in ims.items()};h,w=ims[1].shape[:2]
meta={b:json.loads((root/f'band_{b}'/(pos+'_region_localization')/'localization_metadata.json').read_text()) for b in args.bands};boxes={b:m['common_T1_rectangle_xyxy'].copy() for b,m in meta.items()};order=sorted(boxes,key=lambda b:boxes[b][1]);assert all(boxes[a][3]<=boxes[b][1] for a,b in zip(order,order[1:])), 'Disjoint regions required; overlapping hypotheses need separate review'
native={b:{t:np.array(m['matrices_T1_to_native'][f'T{t}']) for t in range(1,5)} for b,m in meta.items()}
for t in range(1,5):
 digest=hashlib.sha256(paths[t].read_bytes()).hexdigest();assert all(m['input_sha256'][f'T{t}']==digest for m in meta.values())
crop={};polygons={};mask=np.zeros((h,w),np.uint8);status=np.zeros((h,w),np.uint8)
for b in order:
 x0,y0,x1,y1=boxes[b];g=u.grid(y1-y0,x1-x0)+[x0,y0];crop[b]={};polygons[b]={};mask[y0:y1,x0:x1]=255;status[y0:y1,x0:x1]=128 if meta[b]['candidate_region_manual_review'] else 255
 for t in range(1,5):
  coords=u.project(g,native[b][t]);assert u.inside(coords,wh[t]).all();singular=np.linalg.svd(native[b][t][:2,:2],compute_uv=False);np.testing.assert_allclose(singular[0],singular[1],rtol=1e-10)
  rgb=u.remap(ims[t],coords);crop[b][t]=rgb;name=f'T{t}_region_{b}_RGB.png';Image.fromarray(rgb).save(out/name,dpi=(600,600));np.testing.assert_array_equal(np.array(Image.open(out/name)),rgb)
  polygons[b][f'T{t}']=u.project(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),native[b][t]).tolist()
 np.testing.assert_array_equal(crop[b][1],ims[1][y0:y1,x0:x1])
Image.fromarray(mask).save(out/'T1_geometric_observation_mask.png');Image.fromarray(status).save(out/'T1_region_review_status_mask.png')
common_x0=max(box[0] for box in boxes.values());common_x1=min(box[2] for box in boxes.values());assert common_x1-common_x0>w*.7,'Do not lose edge roots through an unjustified narrow join'
gap=24;joined={};sep=[];yy=0
for i,b in enumerate(order):
 if i:sep.append([yy,yy+gap]);yy+=gap
 yy+=boxes[b][3]-boxes[b][1]
for t in range(1,5):
 arrays=[]
 for i,b in enumerate(order):
  if i:arrays.append(np.full((gap,common_x1-common_x0,3),255,np.uint8))
  arrays.append(crop[b][t][:,common_x0-boxes[b][0]:common_x1-boxes[b][0]])
 joined[t]=np.concatenate(arrays,axis=0);Image.fromarray(joined[t]).save(out/f'T{t}_multiregion_RGB.png',dpi=(600,600))
jm=np.full(joined[1].shape[:2],255,np.uint8)
for y0,y1 in sep:jm[y0:y1]=0
Image.fromarray(jm).save(out/'joined_geometric_observation_mask.png')
font=lambda n:ImageFont.truetype(u.FONT,n);phases={1:'苗期',2:'抽穗期',3:'成熟期',4:'枯黄期'};colours={'A':'#91d7e4','B':'#edbe82','C':'#b8b5dd'}
pw=1600;ph=max(round(a.shape[0]*pw/a.shape[1]) for a in ims.values());head=145;foot=120;canvas=Image.new('RGB',(4*pw,head+ph+foot),'white');d=ImageDraw.Draw(canvas)
for t in range(1,5):
 ox=(t-1)*pw;im=Image.fromarray(ims[t]);im=im.resize((pw,round(im.height*pw/im.width)),Image.Resampling.LANCZOS);canvas.paste(im,(ox,head));d.text((ox+20,12),f'{pos} | T{t} {phases[t]}',font=font(35),fill='#24343c');d.text((ox+20,72),'按原始T1定位；分别核对不同偏移区域',font=font(27),fill='#52626d')
 for b in order:
  pp=np.array(polygons[b][f'T{t}'])*pw/wh[t][0]+[ox,head];pp=[tuple(q) for q in pp];d.line(pp+[pp[0]],fill=colours[b],width=3);top=min(q[1] for q in pp);y=min(max(top+5,head+5),head+im.height-49);label=b+('候选' if meta[b]['candidate_region_manual_review'] else '')
  d.rectangle((ox+18,y,ox+18+len(label)*35,y+45),fill='white');d.text((ox+23,y-2),label,font=font(30),fill='#24343c')
 d.text((ox+20,head+ph+14),'原色；每段整体定位，保留根系局部变化',font=font(26),fill='#52626d');d.text((ox+20,head+ph+63),'框为几何范围；候选区域仍需人工复核',font=font(26),fill='#52626d')
canvas.save(out/f'{pos}_reviewed_region_localization.png',dpi=(600,600));canvas.resize((2600,round(canvas.height*2600/canvas.width)),Image.Resampling.LANCZOS).save(out/f'{pos}_region_preview.jpg',quality=97,subsampling=0)
pw=1000;rh=round(joined[1].shape[0]*pw/joined[1].shape[1]);sequence=Image.new('RGB',(4*pw,rh+185),'white');d=ImageDraw.Draw(sequence)
for t in range(1,5):
 ox=(t-1)*pw;sequence.paste(Image.fromarray(joined[t]).resize((pw,rh),Image.Resampling.LANCZOS),(ox,110));d.text((ox+15,12),f'T{t} {phases[t]}',font=font(35),fill='#24343c');d.text((ox+15,64),'顺序：'+ '/'.join(order)+'；白带为非数据分隔',font=font(24),fill='#52626d');d.text((ox+15,110+rh+15),'同一观察区域；接缝不可用于根系连续追踪',font=font(23),fill='#52626d')
sequence.save(out/f'{pos}_multiregion_sequence.png',dpi=(600,600))
windows=json.loads(args.windows.read_text()) if args.windows else [dict(band=b,label=f'{b}区局部',xyxy=[boxes[b][0]+(boxes[b][2]-boxes[b][0])//4,boxes[b][1],boxes[b][0]+3*(boxes[b][2]-boxes[b][0])//4,boxes[b][3]]) for b in order]
rows=[];pw=1000
for v in windows:
 b=v['band'];x0,y0,x1,y1=v['xyxy'];bx,by,bxx,byy=boxes[b];assert bx<=x0<x1<=bxx and by<=y0<y1<=byy;rh=round((y1-y0)*pw/(x1-x0));row=Image.new('RGB',(4*pw,rh+165),'white');d=ImageDraw.Draw(row)
 for t in range(1,5):
  rgb=crop[b][t][y0-by:y1-by,x0-bx:x1-bx];row.paste(Image.fromarray(rgb).resize((pw,rh),Image.Resampling.LANCZOS),((t-1)*pw,110));d.text(((t-1)*pw+15,12),f'{v["label"]} | T{t}'+(' | 候选' if meta[b]['candidate_region_manual_review'] else ''),font=font(29),fill='#24343c');d.text(((t-1)*pw+15,63),'同一T1观察窗；原色；保留局部位移',font=font(24),fill='#52626d');d.text(((t-1)*pw+15,110+rh+10),'对照根系和土层结构；不据此判定根系身份',font=font(22),fill='#52626d')
 rows.append(row)
review=Image.new('RGB',(4*pw,sum(a.height for a in rows)),'white');yy=0
for row in rows:review.paste(row,(0,yy));yy+=row.height
review.save(out/f'{pos}_root_region_review.png',dpi=(600,600));review.resize((2200,round(review.height*2200/review.width)),Image.Resampling.LANCZOS).save(out/f'{pos}_root_review_preview.jpg',quality=97,subsampling=0)
neural_checks={};extra=root/pos/'T1_to_T4/sampled_matches.npz'
if extra.exists():
 with np.load(extra) as z:pa=z['source_xy'][z['reliable']]*1600/wh[1][0];pb=z['target_xy'][z['reliable']]*1600/wh[4][0]
 left=np.arange(len(pa));modes=[]
 for _ in range(3):
  if len(left)<6:break
  E,k=cv2.estimateAffinePartial2D(pa[left].astype('float32'),pb[left].astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999)
  if E is None:break
  ii=left[k.ravel()>0];E=np.vstack([E,[0,0,1]]);scale=np.hypot(E[0,0],E[1,0]);angle=np.degrees(np.arctan2(E[1,0],E[0,0]));left=np.setdiff1d(left,ii)
  if len(ii)<20 or not (.8<scale<1.2 and abs(angle)<10):continue
  N=np.diag([wh[4][0]/1600]*2+[1])@E@np.diag([1600/wh[1][0]]*2+[1]);comparisons={}
  for b,(x0,y0,x1,y1) in boxes.items():
   g=u.grid(y1-y0,x1-x0)[::16,::16]+[x0,y0];err=np.linalg.norm(u.project(g,N)-u.project(g,native[b][4]),axis=-1)*wh[1][0]/wh[4][0];comparisons[b]=dict(median_T1_px=float(np.median(err)),p95_T1_px=float(np.percentile(err,95)))
  modes.append(dict(consensus_samples=len(ii),matrix_analysis1600=E.tolist(),region_comparisons=comparisons))
 neural_checks=dict(pair='T1_to_T4',bidirectionally_supported_candidates=len(pa),modes=modes,interpretation='Additional full-frame RoMa proposal modes only; internal confidence and model comparison are not ground truth. One dominant mode cannot exclude another supported observation region.')
audit=dict(position=pos,reference='Original T1 pixel centres and scale',regions_T1_xyxy=boxes,region_RGB_wh={b:[boxes[b][2]-boxes[b][0],boxes[b][3]-boxes[b][1]] for b in order},order=order,region_manual_review={b:meta[b]['candidate_region_manual_review'] for b in order},observed_source_area_pct_T1=float(100*np.count_nonzero(mask)/mask.size),region_route_pass_geometric_pct_T1=float(100*np.count_nonzero(status==255)/status.size),joined_RGB_wh=list(joined[1].shape[1::-1]),joined_T1_x=[common_x0,common_x1],nondata_separator_rows=sep,polygons_native=polygons,root_observation_windows=windows,band_metadata=meta,additional_neural_region_checks=neural_checks,input_sha256=meta[order[0]]['input_sha256'],validation=dict(source_hashes_checked=True,T1_crops_pixel_exact=True,source_pixel_bounds_checked=True,RGB_encoding_checked=True,uniform_global_similarities_only=True),limitations='Area is geometric observation, including explicitly labelled candidates; not confident root/point support. All descriptor cross-validation and location fits are conditional on full-frame and region hypotheses, not independent manual TRE. All route disagreements are retained. White separators omit unobserved T1 gaps and cannot establish root continuity. No device-period mechanism, root identities, new-root/death labels, colour correction, elastic deformation, or invented detail is asserted.')
(root/'multiregion_review_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
primary=[out/f'{pos}_reviewed_region_localization.png',out/f'{pos}_root_region_review.png',out/f'{pos}_multiregion_sequence.png',out/'T1_geometric_observation_mask.png',out/'T1_region_review_status_mask.png']+[out/f'T{t}_region_{b}_RGB.png' for b in order for t in range(1,5)]
secondary=[out/f'T{t}_multiregion_RGB.png' for t in range(1,5)]+[out/'joined_geometric_observation_mask.png']
for suffix,files in [('reviewed_regions_and_RGB',primary),('four_stitched_RGB',secondary)]:
 archive=root/f'{pos}_{suffix}.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in files:z.write(f,f.name)
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None and all(n.endswith('.png') for n in z.namelist())
 assert archive.stat().st_size<100*1024**2;print('ZIP',archive,archive.stat().st_size,flush=True)
print('RENDER_VALIDATED',pos,json.dumps({k:audit[k] for k in ['regions_T1_xyxy','region_RGB_wh','observed_source_area_pct_T1','region_route_pass_geometric_pct_T1','region_manual_review','joined_RGB_wh']},ensure_ascii=False),flush=True)
