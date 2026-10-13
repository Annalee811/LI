#!/usr/bin/env python3
"""Broad B2 region search: complete RGB canvases, multiscale descriptors, root-guided verification.
Seasonal/unmatched content is not rejected as invalid observational area.
"""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'input/raw/V-变化/A017-1-B2';OUT=ROOT/'B2_region_revision';OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4);cv2.setRNGSeed(20261013);results=[]
for width in [1000,1600]:
 images={t:np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB')) for t in [1,2,4]}
 resized={t:cv2.resize(a,(width,round(a.shape[0]*width/a.shape[1])),interpolation=cv2.INTER_AREA if a.shape[1]>width else cv2.INTER_CUBIC) for t,a in images.items()}
 for mode in ['raw','clahe','blur','ridge']:
  desc={}
  for t,a in resized.items():
   g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
   if mode=='blur':g=cv2.GaussianBlur(g,(0,0),1.6*width/1600)
   if mode!='raw':g=cv2.createCLAHE(2,(12,12)).apply(g)
   if mode=='ridge':g=cv2.normalize(cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))),None,0,255,cv2.NORM_MINMAX).astype('uint8')
   k,d=cv2.SIFT_create(nfeatures=22000,contrastThreshold=.006,edgeThreshold=18).detectAndCompute(g,None);d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-8));desc[t]=(k,d)
  for a,b in [(1,2),(2,4),(1,4)]:
   ka,da=desc[a];kb,db=desc[b];bf=cv2.BFMatcher();ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2);rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.85*n.distance};ms=[m for m,n in ab if m.distance<.85*n.distance and rev.get(m.trainIdx)==m.queryIdx]
   pa=np.array([ka[m.queryIdx].pt for m in ms],np.float32)*1600/width;pb=np.array([kb[m.trainIdx].pt for m in ms],np.float32)*1600/width
   row=dict(pair=f'T{a}_to_T{b}',analysis_width=width,mode=mode,mutual_matches=len(ms),source_points_analysis_1600=pa.tolist(),target_points_analysis_1600=pb.tolist(),hypotheses=[])
   left=np.arange(len(ms))
   # Keep competing hypotheses: repetitive soil must not silently choose a neighboring band.
   for candidate in range(3):
    if len(left)<6:break
    H,ok=cv2.estimateAffinePartial2D(pa[left],pb[left],method=cv2.RANSAC,ransacReprojThreshold=10,maxIters=50000,confidence=.999,refineIters=20)
    if H is None:break
    ids=left[ok.ravel()>0];H=np.vstack([H,[0,0,1]]);scale=np.hypot(H[0,0],H[1,0]);angle=np.degrees(np.arctan2(H[1,0],H[0,0]));error=np.linalg.norm(cv2.perspectiveTransform(pa[ids,None],H)[:,0]-pb[ids],axis=-1)
    row['hypotheses'].append(dict(matrix_analysis_1600=H.tolist(),inliers=int(len(ids)),inlier_indices=ids.tolist(),normalized_scale=float(scale),angle_deg=float(angle),median_fit_analysis_px=float(np.median(error)),source_span_xy=np.ptp(pa[ids],axis=0).tolist()))
    left=np.setdiff1d(left,ids)
   results.append(row);print(json.dumps({k:v for k,v in row.items() if 'points' not in k}),flush=True)
(OUT/'full_canvas_feature_search.json').write_text(json.dumps(results,indent=2))
