"""Root-guided multi-channel matching; uniform scale coordinates; preserve shape."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'input/raw/V-变化/A017-1-B2';OUT=ROOT/'B2_shape_revision';OUT.mkdir(exist_ok=True)
cv2.setNumThreads(4);cv2.setRNGSeed(20261013)
W=1600
bands={1:(.38,.83),2:(0,.47),4:(.65,1)}
ims={t:np.array(Image.open(RAW/f'T{t}.jpg').convert('RGB')) for t in bands}
small={t:cv2.resize(a,(W,round(a.shape[0]*W/a.shape[1])),interpolation=cv2.INTER_AREA if a.shape[1]>W else cv2.INTER_CUBIC) for t,a in ims.items()}
results=[]
for mode in ['raw','clahe','blur','ridge']:
 desc={}
 for t,a in small.items():
  g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
  if mode=='blur':g=cv2.GaussianBlur(g,(0,0),1.6)
  if mode!='raw':g=cv2.createCLAHE(2,(12,12)).apply(g)
  if mode=='ridge':g=cv2.normalize(cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))),None,0,255,cv2.NORM_MINMAX).astype('uint8')
  mask=np.zeros(g.shape,np.uint8);lo,hi=bands[t];mask[round(lo*g.shape[0]):round(hi*g.shape[0])]=255
  k,d=cv2.SIFT_create(nfeatures=18000,contrastThreshold=.006,edgeThreshold=18).detectAndCompute(g,mask);d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-8));desc[t]=(k,d)
 for ta,tb in [(1,2),(2,4),(1,4)]:
  ka,da=desc[ta];kb,db=desc[tb];bf=cv2.BFMatcher();ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2)
  rev={m.queryIdx:m.trainIdx for m,n in ba if m.distance<.85*n.distance}
  ms=[m for m,n in ab if m.distance<.85*n.distance and rev.get(m.trainIdx)==m.queryIdx]
  pa=np.array([ka[m.queryIdx].pt for m in ms],np.float32);pb=np.array([kb[m.trainIdx].pt for m in ms],np.float32)
  r=dict(mode=mode,pair=f'T{ta}_to_T{tb}',matches=len(ms))
  if len(ms)>=4:
   H,ok=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=5,maxIters=50000,confidence=.999,refineIters=20);ok=ok.ravel()>0;H=np.vstack([H,[0,0,1]])
   err=np.linalg.norm(cv2.perspectiveTransform(pa[:,None],H)[:,0]-pb,axis=1)
   r.update(matrix_analysis=H.tolist(),inliers=int(ok.sum()),median=float(np.median(err[ok])),p95=float(np.percentile(err[ok],95)),scale=float(np.hypot(H[0,0],H[0,1])),angle=float(np.degrees(np.arctan2(H[1,0],H[0,0]))),source_points=pa.tolist(),target_points=pb.tolist(),inlier_flags=ok.tolist())
   canvas=np.full((max(small[ta].shape[0],small[tb].shape[0]),W*2,3),255,np.uint8);canvas[:small[ta].shape[0],:W]=small[ta];canvas[:small[tb].shape[0],W:]=small[tb]
   for i in np.flatnonzero(ok):
    p=tuple(pa[i].round().astype(int));q=tuple((pb[i]+[W,0]).round().astype(int));cv2.circle(canvas,p,4,(20,255,150),1);cv2.circle(canvas,q,4,(20,255,150),1);cv2.line(canvas,p,q,(230,220,70),1)
   Image.fromarray(canvas).save(OUT/f'{mode}_{r["pair"]}_feature_correspondences.jpg',quality=95)
  results.append(r);print(json.dumps({k:v for k,v in r.items() if 'points' not in k and k!='inlier_flags'}),flush=True)
(OUT/'shape_feature_candidates.json').write_text(json.dumps(results,indent=2))
