#!/usr/bin/env python3
"""Descriptor verification inside hypothesised available-period observation regions.

This is explicitly conditioned on complete-frame location proposals. It is not
independent ground truth. Original RGB is never changed. All candidate rows and
competing geometries are retained for later validation and route comparison.
"""
from pathlib import Path
import argparse,json
import numpy as np,cv2
from PIL import Image
import B2_region_tools as u

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--position',required=True);p.add_argument('--bands',nargs='+',required=True);p.add_argument('--threads',type=int,default=1);args=p.parse_args()
root=args.root;pos=args.position;W=1600;cv2.setNumThreads(args.threads);cv2.setRNGSeed(20261014)
raw=root/'input/raw/V-变化'/pos;ims={int(f.stem[1:]):np.array(Image.open(f).convert('RGB')) for f in raw.glob('T*')};wh={t:a.shape[1::-1] for t,a in ims.items()};TS=sorted(ims)
small={t:cv2.resize(a,(W,round(a.shape[0]*W/a.shape[1])),interpolation=cv2.INTER_AREA if a.shape[1]>W else cv2.INTER_CUBIC) for t,a in ims.items()}
base=json.loads((root/(pos+'_region_localization')/'full_canvas_feature_search.json').read_text())
for band in args.bands:
 d=root/('band_'+band);prior=json.loads((d/'prior.json').read_text());poses={1:np.eye(3)}
 for a,b in zip(TS,TS[1:]):
  H=np.diag([W/wh[b][0]]*2+[1])@np.array(prior['matrices_native'][f'T{a}_to_T{b}'])@np.diag([wh[a][0]/W]*2+[1]);poses[b]=H@poses[a]
 g=u.grid(*small[1].shape[:2]);geom=np.logical_and.reduce([u.inside(u.project(g,H),small[t].shape[1::-1]) for t,H in poses.items()]);box=u.rectangle(geom);x0,y0,x1,y1=box
 masks={};polys={};rows=base.copy();coarse_corners=np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32)
 for t,H in poses.items():
  poly=u.project(coarse_corners,H);polys[f'T{t}']=poly.tolist();mask=np.zeros(small[t].shape[:2],np.uint8);cv2.fillConvexPoly(mask,np.rint(poly).astype('int32'),255);masks[t]=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(161,161)))
 for mode in ['blur','ridge']:
  desc={}
  for t,a in small.items():
   gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
   if mode=='blur':gray=cv2.GaussianBlur(gray,(0,0),1.6)
   gray=cv2.createCLAHE(2,(12,12)).apply(gray)
   if mode=='ridge':gray=cv2.createCLAHE(2,(12,12)).apply(np.maximum.reduce([cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))) for n in [9,15,23]]))
   kp,dd=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.0035,edgeThreshold=20).detectAndCompute(gray,masks[t]);dd=np.sqrt(dd/(dd.sum(1,keepdims=True)+1e-8)) if dd is not None else np.empty((0,128),np.float32);desc[t]=(kp,dd)
  for ai,a in enumerate(TS):
   for b in TS[ai+1:]:
    ka,da=desc[a];kb,db=desc[b];matches=[]
    if len(da)>=2 and len(db)>=2:
     bf=cv2.BFMatcher();ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2);rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.88*n.distance};matches=[m for m,n in ab if m.distance<.88*n.distance and rev.get(m.trainIdx)==m.queryIdx]
    pa=np.array([ka[m.queryIdx].pt for m in matches],np.float32).reshape(-1,2);pb=np.array([kb[m.trainIdx].pt for m in matches],np.float32).reshape(-1,2);left=np.arange(len(matches));hyp=[]
    for _ in range(3):
     if len(left)<6:break
     H,k=cv2.estimateAffinePartial2D(pa[left],pb[left],method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=30000,confidence=.999)
     if H is None:break
     ii=left[k.ravel()>0];H=np.vstack([H,[0,0,1]]);hyp.append(dict(matrix_analysis_1600=H.tolist(),inliers=len(ii),normalized_scale=float(np.hypot(H[0,0],H[1,0])),angle_deg=float(np.degrees(np.arctan2(H[1,0],H[0,0])))));left=np.setdiff1d(left,ii)
    rows.append(dict(pair=f'T{a}_to_T{b}',mode=f'conditioned_{band}_{mode}',analysis_width=W,mutual_matches=len(matches),source_points_analysis_1600=pa.tolist(),target_points_analysis_1600=pb.tolist(),hypotheses=hyp,conditioned_on_region_prior=True,mask_margin_analysis1600_px=80,descriptor_ratio=.88))
    print('CONDITIONED',pos,band,mode,f'{a}-{b}',len(matches),[(h['inliers'],round(h['matrix_analysis_1600'][1][2])) for h in hyp],flush=True)
 out=d/(pos+'_region_localization');(out/'full_canvas_feature_search.json').write_text(json.dumps(rows,indent=2));(d/'conditioned_search_provenance.json').write_text(json.dumps(dict(coarse_T1_rectangle_analysis1600=box,source_polygons_analysis1600=polys,mask_expansion_px=80,interpretation='Descriptor localisation and validation are conditional on full-frame branch hypotheses; not manual TRE or independent absolute accuracy.'),indent=2))
 print('CONDITIONED_DONE',pos,band,flush=True)
