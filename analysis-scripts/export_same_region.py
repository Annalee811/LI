#!/usr/bin/env python3
"""Same-region outline figure + separate equal-size RGB crops.
Uses hash-verified cached RoMa candidates for region hypotheses; independent complete-frame
RootSIFT verifies the geometry. Only global similarity is used to sample original RGB once.
"""
from pathlib import Path
import argparse,json,csv,hashlib,zipfile,math
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import B2_region_tools as u
from run_0353_1_root_priority import support
cli=argparse.ArgumentParser();cli.add_argument('--position',required=True);cli.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);cli.add_argument('--threads',type=int,default=1);args=cli.parse_args();POSITION=args.position
ROOT=args.root.resolve();RAW=ROOT/'input/raw/V-变化'/POSITION;OUT=ROOT/(POSITION+'_region_localization');RESULT=OUT/'images';RESULT.mkdir(exist_ok=True)
cv2.setNumThreads(args.threads);cv2.setRNGSeed(20261013);W=1600;paths={int(f.stem[1:]):f for f in RAW.glob('T*') if f.is_file()};TS=sorted(paths)
if len(TS)<=2:print('SKIPPED <=2 periods',POSITION,flush=True);raise SystemExit(0)
assert TS[0]==1,'T1 reference is required';ims={t:np.array(Image.open(paths[t]).convert('RGB')) for t in TS};wh={t:ims[t].shape[1::-1] for t in TS};manifest=json.loads((ROOT/POSITION/'manifest.json').read_text());rows=json.loads((OUT/'full_canvas_feature_search.json').read_text())
for t in TS:
 key=next(k for k in manifest['input_sha256'] if Path(k).stem==f'T{t}');expected=manifest['input_sha256'][key];expected=expected['sha256'] if isinstance(expected,dict) else expected;assert hashlib.sha256(paths[t].read_bytes()).hexdigest()==expected
root_masks={t:support(ims[t])[0] for t in TS};Hs={};summaries=[];prior_poses={1:np.eye(3)};roma_counts={}
for a,b in zip(TS,TS[1:]):
 name=f'T{a}_to_T{b}';pair=ROOT/POSITION/name
 with np.load(pair/'sampled_matches.npz') as z:
  ok=z['reliable'];pa=z['source_xy'][ok]*W/wh[a][0];pb=z['target_xy'][ok]*W/wh[b][0]
 assert len(pa)>=100,(name,'Cached RoMa regional hint too weak',len(pa));H,mask=cv2.estimateAffinePartial2D(pa.astype('float32'),pb.astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999,refineIters=20);H=np.vstack([H,[0,0,1]]);assert mask.sum()>=100;prior_poses[b]=H@prior_poses[a];roma_counts[name]=dict(reliable_samples=len(pa),global_similarity_consensus=int(mask.sum()))
for i,a in enumerate(TS):
 for b in TS[i+1:]:
  name=f'T{a}_to_T{b}';prior=prior_poses[b]@np.linalg.inv(prior_poses[a]);pa=[];pb=[];seen=set()
  for row in rows:
   if row['pair']!=name:continue
   for p,q in zip(row['source_points_analysis_1600'],row['target_points_analysis_1600']):
    p,q=np.array(p),np.array(q);key=tuple(np.floor(p/4).astype(int))
    if key in seen or np.linalg.norm(u.project(p,prior)-q)>75:continue
    seen.add(key);pa.append(p);pb.append(q)
  if len(pa)<6:
   summaries.append(dict(pair=name,candidates=len(pa),stable_anchors=0,validated_region_edge=False,reason='Insufficient descriptor evidence',manual_TRE=None));continue
  pa,pb=np.array(pa),np.array(pb);root=[]
  for p,q in zip(pa,pb):
   p=np.rint(p*1400/W).astype(int);q=np.rint(q*1400/W).astype(int);root.append(bool(root_masks[a][min(max(p[1],0),root_masks[a].shape[0]-1),min(max(p[0],0),1399)] and root_masks[b][min(max(q[1],0),root_masks[b].shape[0]-1),min(max(q[0],0),1399)]))
  root=np.array(root);cells=np.floor(pa/90).astype(int);_,inv,counts=np.unique(cells,axis=0,return_inverse=True,return_counts=True);weight=np.where(root,3.,1.)/np.sqrt(counts[inv]);H,k=cv2.estimateAffinePartial2D(pa.astype('float32'),pb.astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999,refineIters=20);stable=k.ravel()>0;H=u.fit(pa[stable],pb[stable],weight[stable]);error=np.linalg.norm(u.project(pa,H)-pb,axis=1);stable=error<=10;H=u.fit(pa[stable],pb[stable],weight[stable]);Hs[name]=H
  folds=(cells[:,0]+3*cells[:,1])%4;cv=np.full(len(pa),np.nan);pose_errors=[]
  lo,hi=pa[stable].min(0),pa[stable].max(0);pose_grid=np.stack(np.meshgrid(np.linspace(lo[0],hi[0],9),np.linspace(lo[1],hi[1],9)),axis=-1)
  for fold in range(4):
   train=(folds!=fold)&stable;test=folds==fold
   if train.sum()>=6 and test.any():
    fold_H=u.fit(pa[train],pb[train],weight[train]);cv[test]=np.linalg.norm(u.project(pa[test],fold_H)-pb[test],axis=1)*wh[1][0]/W;pose_errors.extend((np.linalg.norm(u.project(pose_grid,fold_H)-u.project(pose_grid,H),axis=-1)*wh[1][0]/W).ravel())
  good=cv[stable&np.isfinite(cv)];median=float(np.median(good)) if len(good) else None;scale=np.hypot(H[0,0],H[1,0]);angle=np.degrees(np.arctan2(H[1,0],H[0,0]));assert .8<scale<1.2 and abs(angle)<10
  span=np.ptp(pa[stable],axis=0)*wh[1][0]/W
  point_valid=bool(stable.sum()>=20 and median is not None and median<=3 and span[0]>=wh[1][0]*.3)
  pose_median=float(np.median(pose_errors));pose_p95=float(np.percentile(pose_errors,95))
  # Region localization is assessed separately from unchanged-point precision.
  # Local root/soil motion is preserved rather than removed with a nonrigid warp.
  valid=bool(stable.sum()>=20 and median is not None and median<=10 and span[0]>=wh[1][0]*.3 and pose_median<=3 and pose_p95<=5)
  d=dict(pair=name,candidates=len(pa),stable_anchors=int(stable.sum()),root_support_anchor_candidates=int((root&stable).sum()),spatial_CV_anchor_median_T1_px=median,spatial_CV_anchor_p95_T1_px=float(np.percentile(good,95)) if len(good) else None,stable_anchor_span_T1_xy=span.tolist(),matrix_analysis_1600=H.tolist(),scale=float(scale),rotation_deg=float(angle),validated_region_edge=valid,point_precision_3px_pass=point_valid,spatial_fold_pose_median_T1_px=pose_median,spatial_fold_pose_p95_T1_px=pose_p95,manual_TRE=None);summaries.append(d);print('MATCH',POSITION,json.dumps(d),flush=True)
  with (OUT/(name+'_automatic_evidence.csv')).open('w') as f:
   wr=csv.writer(f,lineterminator="\n");wr.writerow(['source_x_analysis1600','source_y_analysis1600','target_x_analysis1600','target_y_analysis1600','stable','root_proxy','spatial_CV_T1_px'])
   for p,q,s,rt,c in zip(pa,pb,stable,root,cv):wr.writerow([*p,*q,int(s),int(rt),float(c)])
lookup={d['pair']:d for d in summaries};adj=[f'T{a}_to_T{b}' for a,b in zip(TS,TS[1:])]
# Prefer the chronological chain when all its edges have independent structural support.
if all(lookup[n].get('point_precision_3px_pass',False) and lookup[n].get('validated_region_edge',False) for n in adj):
 analysis={1:np.eye(3)}
 for a,b in zip(TS,TS[1:]):analysis[b]=Hs[f'T{a}_to_T{b}']@analysis[a]
 selected=adj.copy();method='Validated consecutive-period chain on original T1 coordinates'
else:
 # Build a tree of well-supported structural edges; do not call weak adjacent fits reliable.
 analysis={1:np.eye(3)};selected=[]
 while len(analysis)<len(TS):
  candidates=[]
  for d in summaries:
   if not d.get('validated_region_edge',False):continue
   a,b=[int(s[1:]) for s in d['pair'].split('_to_')]
   if (a in analysis)==(b in analysis):continue
   score=math.log1p(d['stable_anchors'])/max(d['spatial_CV_anchor_median_T1_px'],.5);candidates.append((score,d['pair'],a,b))
  assert candidates,'ADAPTIVE_SEARCH_REQUIRED: reliable structural graph is disconnected'
  _,name,a,b=max(candidates)
  if a in analysis:analysis[b]=Hs[name]@analysis[a]
  else:analysis[a]=np.linalg.inv(Hs[name])@analysis[b]
  selected.append(name)
 method='Spatially stable region structure graph; T1 coordinate reference; point-level precision and regional stability evaluated separately'
native={t:np.diag([wh[t][0]/W]*2+[1])@H@np.diag([W/wh[1][0]]*2+[1]) for t,H in analysis.items()};h,w=ims[1].shape[:2];g=u.grid(h,w);maps={t:u.project(g,H) for t,H in native.items()};geoms={t:u.inside(maps[t],wh[t]) for t in TS};common=np.logical_and.reduce(list(geoms.values()));box=u.rectangle(common);x0,y0,x1,y1=box;sl=np.s_[y0:y1,x0:x1];route_checks={}
for d in summaries:
 name=d['pair']
 if name in selected or not d.get('validated_region_edge',False):continue
 a,b=[int(s[1:]) for s in name.split('_to_')];points=g[sl][::16,::16]*W/w;error=np.linalg.norm(u.project(points,Hs[name]@analysis[a])-u.project(points,analysis[b]),axis=-1)*w/W;route_checks[name]=dict(median_T1_px=float(np.median(error)),p95_T1_px=float(np.percentile(error,95)))
assert route_checks,'ADAPTIVE_SEARCH_REQUIRED: no independent route corroboration'
assert any(d['median_T1_px']<=5 for d in route_checks.values()),('ADAPTIVE_SEARCH_REQUIRED: no consistent independent route',route_checks)
for d in route_checks.values():d['region_route_5px_pass']=bool(d['median_T1_px']<=5)
weak_routes=[n for n,d in route_checks.items() if not d['region_route_5px_pass']]
for t in TS:
 coords=maps[t][sl];a=u.remap(ims[t],coords);Image.fromarray(a).save(RESULT/f'T{t}_common_region_RGB.png',dpi=(600,600));np.testing.assert_array_equal(np.array(Image.open(RESULT/f'T{t}_common_region_RGB.png')),u.remap(ims[t],coords));s=np.linalg.svd(native[t][:2,:2],compute_uv=False);np.testing.assert_allclose(s[0],s[1],rtol=1e-10)
assert common[sl].all()
panelw=1600;panelh=max(round(ims[t].shape[0]*panelw/ims[t].shape[1]) for t in TS);header=145;footer=105;canvas=Image.new('RGB',(len(TS)*panelw,panelh+header+footer),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(u.FONT,38);small=ImageFont.truetype(u.FONT,25);phases={1:'苗期',2:'抽穗期',3:'成熟期',4:'枯黄期'};polygons={}
for k,t in enumerate(TS):
 im=Image.fromarray(ims[t]);im=im.resize((panelw,round(im.height*panelw/im.width)),Image.Resampling.LANCZOS);canvas.paste(im,(k*panelw,header));corners=u.project(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),native[t]);polygons[f'T{t}']=corners.tolist();corners*=panelw/wh[t][0];points=[(float(p[0]+k*panelw),float(p[1]+header)) for p in corners];draw.line(points+[points[0]],fill='#9bd4df',width=2);draw.text((k*panelw+18,12),f'T{t} {phases[t]}：同一区域定位',font=font,fill='#24343c');draw.text((k*panelw+18,75),'原色原图；细框为共同观察区域',font=small,fill='#52626d');draw.text((k*panelw+18,header+panelh+12),f'独立裁剪图：{x1-x0}×{y1-y0} T1像素',font=small,fill='#52626d');draw.text((k*panelw+18,header+panelh+56),'自动区域定位；保留根系及土层局部变化',font=small,fill='#52626d')
name=POSITION+'_same_region_localization.png';canvas.save(RESULT/name,dpi=(600,600));canvas.resize((2200,round(canvas.height*2200/canvas.width)),Image.Resampling.LANCZOS).save(RESULT/(POSITION+'_same_region_preview.jpg'),quality=96,subsampling=0)
meta=dict(position=POSITION,period_ids=TS,missing_period_ids=[t for t in range(1,5) if t not in TS],reference='Original T1 zero-based pixel centres',common_T1_rectangle_xyxy=box,crop_size_wh=[x1-x0,y1-y0],common_observable_pct_T1=100*common.mean(),matrices_T1_to_native={f'T{t}':H.tolist() for t,H in native.items()},original_canvas_region_polygons=polygons,automatic_feature_checks=summaries,independent_route_checks=route_checks,selected_structure_edges=selected,independent_route_review_required=weak_routes,regional_stability_rule=dict(minimum_consensus_anchors=20,minimum_horizontal_span_fraction=.3,spatial_fold_pose_median_T1_px_max=3,spatial_fold_pose_p95_T1_px_max=5,point_CV_median_T1_px_max=10,independent_route_at_least_one_median_T1_px_max=5),cached_RoMa_consensus=roma_counts,manual_landmark_TRE=None,method=method,rendering='Original-canvas outline figure uses original RGB without elastic deformation or colour correction; separate crops use a single GLOBAL similarity resampling on T1 scale.',validation_limits='Automatic spatial CV conditional on selected descriptor-consensus anchors, not human landmarks. Pose-fold stability quantifies sensitivity to blocks of selected anchors, not absolute accuracy. Independent route disagreements remain recorded and require visual review; no exact point-registration success is inferred. New roots and changed soil remain in the crop. Upsampling/DPI does not create captured detail.',input_sha256={f'T{t}':hashlib.sha256(paths[t].read_bytes()).hexdigest() for t in TS})
(OUT/'localization_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));zip_path=ROOT.parent/(POSITION+'_same_region_map_and_four_RGB.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 z.write(RESULT/name,name)
 for t in TS:z.write(RESULT/f'T{t}_common_region_RGB.png',f'T{t}_common_region_RGB.png')
with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None;assert len(z.namelist())==len(TS)+1
print('PASS',POSITION,'same-region original frames and equal-size single-resample RGB crops',box,[x1-x0,y1-y0],method,'routes',route_checks,flush=True);print('ZIP',zip_path,zip_path.stat().st_size,flush=True)
