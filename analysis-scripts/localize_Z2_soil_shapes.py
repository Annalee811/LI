"""Locate Z2's stable lower-soil crack shapes on normalized analysis images."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path.cwd()/'results/microtube_registration';RAW=ROOT/'input/raw/V-变化/A017-1-Z2'
OUT=ROOT/'A017-1_Z2_soil_shapes';OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4);cv2.setRNGSeed(20261013)

def prepare(t,mode):
 a=np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB'));h,w=a.shape[:2];size=(1400,round(1400*h/w));rgb=cv2.resize(a,size,interpolation=cv2.INTER_AREA if w>1400 else cv2.INTER_CUBIC)
 g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
 if mode=='blur_clahe':g=cv2.GaussianBlur(g,(0,0),3.0)
 g=cv2.createCLAHE(2,(12,12)).apply(g)
 if mode=='crack':g=cv2.normalize(cv2.morphologyEx(g,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(35,35))),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
 mask=np.zeros_like(g);lo,hi=(.58,1) if t in [1,2] else (0,.44);mask[round(g.shape[0]*lo):round(g.shape[0]*hi)]=255
 keys,d=cv2.SIFT_create(nfeatures=10000,contrastThreshold=.012,edgeThreshold=14).detectAndCompute(g,mask)
 d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-7))
 return keys,d,np.array([w/size[0],h/size[1]]),rgb

all_results=[]
for mode in ['clahe','blur_clahe','crack']:
 prepared={t:prepare(t,mode) for t in [1,2,4]}
 for a,b in [(1,2),(2,4),(1,4)]:
  ka,da,sa,ia=prepared[a];kb,db,sb,ib=prepared[b]
  match=cv2.BFMatcher();ab=match.knnMatch(da,db,2);ba=match.knnMatch(db,da,2)
  rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.8*n.distance};ms=[m for m,n in ab if m.distance<.8*n.distance and rev.get(m.trainIdx)==m.queryIdx]
  pa=np.array([ka[m.queryIdx].pt for m in ms],np.float32);pb=np.array([kb[m.trainIdx].pt for m in ms],np.float32)
  row=dict(mode=mode,pair=f'T{a}_to_T{b}',source_period=a,target_period=b,mutual_matches=len(ms))
  if len(ms)>=6:
   mat,mask=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=30000,confidence=.999,refineIters=10);ok=mask.ravel()>0
   errors=np.linalg.norm(pa@mat[:,:2].T+mat[:,2]-pb,axis=1)
   row.update(inliers=int(ok.sum()),source_spread_xy=np.ptp(pa[ok],axis=0).tolist(),median_fit_error_analysis_px=float(np.median(errors[ok])),matrix_analysis=np.vstack([mat,[0,0,1]]).tolist(),matrix_native=(np.diag([*sb,1])@np.vstack([mat,[0,0,1]])@np.diag([*(1/sa),1])).tolist(),source_points_analysis=pa[ok].tolist(),target_points_analysis=pb[ok].tolist())
   view=np.full((max(ia.shape[0],ib.shape[0]),2800,3),255,np.uint8);view[:ia.shape[0],:1400]=ia;view[:ib.shape[0],1400:]=ib
   for j in np.flatnonzero(ok):
    p=tuple(np.rint(pa[j]).astype(int));q=tuple(np.rint(pb[j]+[1400,0]).astype(int));cv2.circle(view,p,5,(0,255,100),2);cv2.circle(view,q,5,(0,255,100),2);cv2.line(view,p,q,(230,230,10),1)
   Image.fromarray(view).save(OUT/f'{mode}_T{a}_to_T{b}_matches.png')
  all_results.append(row);print(json.dumps({k:v for k,v in row.items() if 'points' not in k},ensure_ascii=False),flush=True)
(OUT/'soil_shape_candidates.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2))
