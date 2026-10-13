#!/usr/bin/env python3
"""Only same-region localization and four separate equal-size RGB crops for 0353-1-B3.
Use hash-verified cached RoMa candidates to localize the region, verify with full-frame
multiscale RootSIFT, and use GLOBAL similarity only for single-resample RGB.
"""
from pathlib import Path
import json,csv,hashlib,zipfile,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import B2_region_tools as u
from run_0353_1_root_priority import support
ROOT=Path(__file__).resolve().parent;POSITION='0353-1-B3';RAW=ROOT/'input/raw/V-变化'/POSITION;OUT=ROOT/(POSITION+'_region_localization');RESULT=OUT/'images';RESULT.mkdir(exist_ok=True)
cv2.setNumThreads(4);cv2.setRNGSeed(20261013);W=1600;TS=[1,2,3,4];ims={t:np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB')) for t in TS};wh={t:ims[t].shape[1::-1] for t in TS};manifest=json.loads((ROOT/POSITION/'manifest.json').read_text());rows=json.loads((OUT/'full_canvas_feature_search.json').read_text())
for t in TS:
 key=next(k for k in manifest['input_sha256'] if Path(k).name==f'T{t}.jpg');assert hashlib.sha256((RAW/f'T{t}.jpg').read_bytes()).hexdigest()==manifest['input_sha256'][key]
root_masks={t:support(ims[t])[0] for t in TS};Hs={};summaries=[];roma_priors={}
for a,b in [(1,2),(2,3),(3,4)]:
 name=f'T{a}_to_T{b}';pair=ROOT/POSITION/name;z=np.load(pair/'sampled_matches.npz');ok=z['reliable'];pa=z['source_xy'][ok]*W/wh[a][0];pb=z['target_xy'][ok]*W/wh[b][0]
 H,mask=cv2.estimateAffinePartial2D(pa.astype('float32'),pb.astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999,refineIters=20);H=np.vstack([H,[0,0,1]]);assert mask.sum()>=100;roma_priors[name]=H
for a,b in [(1,2),(2,3),(3,4),(1,3),(1,4),(2,4)]:
 name=f'T{a}_to_T{b}'
 if name in roma_priors:prior=roma_priors[name]
 elif (a,b)==(1,3):prior=roma_priors['T2_to_T3']@roma_priors['T1_to_T2']
 elif (a,b)==(1,4):prior=roma_priors['T3_to_T4']@roma_priors['T2_to_T3']@roma_priors['T1_to_T2']
 else:prior=roma_priors['T3_to_T4']@roma_priors['T2_to_T3']
 pa=[];pb=[];seen=set()
 for row in rows:
  if row['pair']!=name:continue
  for p,q in zip(row['source_points_analysis_1600'],row['target_points_analysis_1600']):
   p,q=np.array(p),np.array(q);key=tuple(np.floor(p/4).astype(int))
   if key in seen or np.linalg.norm(u.project(p,prior)-q)>75:continue
   seen.add(key);pa.append(p);pb.append(q)
 pa,pb=np.array(pa),np.array(pb);assert len(pa)>=6,(name,len(pa));root=[]
 for p,q in zip(pa,pb):
  p=np.rint(p*1400/W).astype(int);q=np.rint(q*1400/W).astype(int);root.append(bool(root_masks[a][min(max(p[1],0),root_masks[a].shape[0]-1),min(max(p[0],0),1399)] and root_masks[b][min(max(q[1],0),root_masks[b].shape[0]-1),min(max(q[0],0),1399)]))
 root=np.array(root);cells=np.floor(pa/90).astype(int);_,inv,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True);weight=np.where(root,3.,1.)/np.sqrt(counts[inv])
 H,ok=cv2.estimateAffinePartial2D(pa.astype('float32'),pb.astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999,refineIters=20);stable=ok.ravel()>0;H=u.fit(pa[stable],pb[stable],weight[stable]);error=np.linalg.norm(u.project(pa,H)-pb,axis=1);stable=error<=10;H=u.fit(pa[stable],pb[stable],weight[stable]);Hs[name]=H
 folds=(cells[:,0]+3*cells[:,1])%4;cv=np.full(len(pa),np.nan)
 for fold in range(4):
  train=(folds!=fold)&stable;test=folds==fold
  if train.sum()>=6 and test.any():cv[test]=np.linalg.norm(u.project(pa[test],u.fit(pa[train],pb[train],weight[train]))-pb[test],axis=1)*wh[1][0]/W
 good=cv[stable&np.isfinite(cv)];scale=np.hypot(H[0,0],H[1,0]);angle=np.degrees(np.arctan2(H[1,0],H[0,0]));assert .8<scale<1.2 and abs(angle)<10
 d=dict(pair=name,candidates=len(pa),stable_anchors=int(stable.sum()),root_support_anchor_candidates=int((root&stable).sum()),spatial_CV_anchor_median_T1_px=float(np.median(good)),spatial_CV_anchor_p95_T1_px=float(np.percentile(good,95)),matrix_analysis_1600=H.tolist(),scale=float(scale),rotation_deg=float(angle),manual_TRE=None);summaries.append(d);print('MATCH',json.dumps(d),flush=True)
 with (OUT/(name+'_automatic_evidence.csv')).open('w') as f:
  wr=csv.writer(f);wr.writerow(['source_x_analysis1600','source_y_analysis1600','target_x_analysis1600','target_y_analysis1600','stable','root_proxy','spatial_CV_T1_px'])
  for p,q,s,rt,c in zip(pa,pb,stable,root,cv):wr.writerow([*p,*q,int(s),int(rt),float(c)])
# Adjacent chain establishes the continuous T1 reference. Direct pairs only corroborate location.
analysis={1:np.eye(3)}
for t in [2,3,4]:analysis[t]=Hs[f'T{t-1}_to_T{t}']@analysis[t-1]
native={t:np.diag([wh[t][0]/W]*2+[1])@H@np.diag([W/wh[1][0]]*2+[1]) for t,H in analysis.items()};h,w=ims[1].shape[:2];g=u.grid(h,w);maps={t:u.project(g,H) for t,H in native.items()};geoms={t:u.inside(maps[t],wh[t]) for t in TS};common=np.logical_and.reduce(list(geoms.values()));box=u.rectangle(common);x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];closures={}
for t in [3,4]:closures[f'T1_to_T{t}']=np.linalg.norm(u.project(g[sl][::16,::16]*W/w,analysis[t])-u.project(g[sl][::16,::16]*W/w,Hs[f'T1_to_T{t}']),axis=-1)*w/W
# Region localization needs stable anchors and independent-route agreement, not identical roots.
for d in summaries:
 if d['pair'] in ['T1_to_T2','T2_to_T3','T3_to_T4']:assert d['stable_anchors']>=20 and d['spatial_CV_anchor_median_T1_px']<=3,d
for k,e in closures.items():assert np.median(e)<=5,(k,float(np.median(e)))
for t in TS:
 coords=maps[t][sl];a=u.remap(ims[t],coords);Image.fromarray(a).save(RESULT/f'T{t}_common_region_RGB.png',dpi=(600,600));np.testing.assert_array_equal(np.array(Image.open(RESULT/f'T{t}_common_region_RGB.png')),u.remap(ims[t],coords));s=np.linalg.svd(native[t][:2,:2],compute_uv=False);np.testing.assert_allclose(s[0],s[1],rtol=1e-10)
assert common[sl].all()
# The same rectangle is projected onto each ORIGINAL RGB canvas; originals themselves are not warped.
panelw=1600;panelh=max(round(ims[t].shape[0]*panelw/ims[t].shape[1]) for t in TS);header=145;footer=65;canvas=Image.new('RGB',(4*panelw,panelh+header+footer),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(u.FONT,38);small=ImageFont.truetype(u.FONT,25);phases={1:'苗期',2:'抽穗期',3:'成熟期',4:'枯黄期'};polygons={}
for k,t in enumerate(TS):
 im=Image.fromarray(ims[t]);im=im.resize((panelw,round(im.height*panelw/im.width)),Image.Resampling.LANCZOS);canvas.paste(im,(k*panelw,header));corners=u.project(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),native[t]);polygons[f'T{t}']=corners.tolist();corners*=panelw/wh[t][0];points=[(float(p[0]+k*panelw),float(p[1]+header)) for p in corners];draw.line(points+[points[0]],fill='#9bd4df',width=2);draw.text((k*panelw+18,12),f'T{t} {phases[t]}：同一区域定位',font=font,fill='#24343c');draw.text((k*panelw+18,75),'原色原图；细框为四期共同观察区域',font=small,fill='#52626d');draw.text((k*panelw+18,header+panelh+12),f'独立裁剪图：{x1-x0}×{y1-y0} T1像素',font=small,fill='#52626d')
name='0353-1-B3_same_region_localization.png';canvas.save(RESULT/name,dpi=(600,600));preview=canvas.resize((2200,round(canvas.height*2200/canvas.width)),Image.Resampling.LANCZOS);preview.save(RESULT/'0353-1-B3_same_region_preview.jpg',quality=96,subsampling=0)
meta=dict(position=POSITION,period_ids=TS,missing_period_ids=[],reference='Original T1 zero-based pixel centres',common_T1_rectangle_xyxy=box,crop_size_wh=[x1-x0,y1-y0],common_observable_pct_T1=100*common.mean(),matrices_T1_to_native={f'T{t}':H.tolist() for t,H in native.items()},original_canvas_region_polygons=polygons,automatic_feature_checks=summaries,independent_route_checks={k:dict(median_T1_px=float(np.median(e)),p95_T1_px=float(np.percentile(e,95))) for k,e in closures.items()},manual_landmark_TRE=None,method='Hash-verified cached full RoMa candidate priors + complete-frame multiscale RootSIFT, root/soil weighted robust similarity. Adjacent chain; direct route cross-check. Geometric shared rectangle, not identical-content mask.',rendering='Original-canvas outline figure uses original RGB without elastic deformation or colour correction; separate crops use a single GLOBAL similarity resampling on T1 scale.',validation_limits='Automatic spatial CV is conditional on selected descriptor-consensus anchors, not human landmarks. New roots and changed soil remain in the crop. T2/T3 have lower native resolution; upsampling/DPI does not create captured detail.',input_sha256={f'T{t}':hashlib.sha256((RAW/f'T{t}.jpg').read_bytes()).hexdigest() for t in TS})
(OUT/'localization_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
zip_path=Path('/workspace/LI/results/0353-1-B3_same_region_map_and_four_RGB.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.write(RESULT/name,name)
 for t in TS:z.write(RESULT/f'T{t}_common_region_RGB.png',f'T{t}_common_region_RGB.png')
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None;assert len(z.namelist())==5
print('PASS same-region original frames, four equal-size single-resample RGB crops',box,[x1-x0,y1-y0],flush=True);print('ZIP',zip_path,zip_path.stat().st_size,flush=True)
