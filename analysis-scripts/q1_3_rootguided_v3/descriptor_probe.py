import os
from pathlib import Path
import json,cv2,numpy as np,itertools
from PIL import Image
r=Path(os.environ['LI_Q1_RECHECK_WORK']);m=json.loads((r/'search_priors.json').read_text());cv2.setNumThreads(1);cv2.setRNGSeed(20261019);summary={}
for mode in ['clahe','ridge','blur']:
 ds={}
 for t in [1,2,3]:
  b=m['native_boxes'][str(t)];wh=m['original_wh'][str(t)];im=np.array(Image.open(r/f'T{t}_search_RGB.png'));im=cv2.resize(im,None,fx=1600/wh[0],fy=1600/wh[0]);g=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);g=cv2.createCLAHE(3,(10,10)).apply(g)
  if mode=='ridge':g=np.maximum.reduce([cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))) for n in [7,11,17]])
  if mode=='blur':g=cv2.GaussianBlur(g,(0,0),1)
  k,d=cv2.SIFT_create(nfeatures=14000,contrastThreshold=.002,edgeThreshold=25).detectAndCompute(g,None);d=np.sqrt(d/(d.sum(1,keepdims=True)+1e-9));ds[t]=(np.array([q.pt for q in k])+np.array(b[:2])*1600/wh[0],d)
 for a,b in itertools.combinations([1,2,3],2):
  pa,da=ds[a];pb,db=ds[b];bf=cv2.BFMatcher();ab=bf.knnMatch(da,db,k=2);ba=bf.knnMatch(db,da,k=2);rev={v.queryIdx:v.trainIdx for v,n in ba if v.distance<.9*n.distance};matches=[v for v,n in ab if v.distance<.9*n.distance and rev.get(v.trainIdx)==v.queryIdx];p=np.array([pa[v.queryIdx] for v in matches]);q=np.array([pb[v.trainIdx] for v in matches]);left=np.arange(len(p));fits=[]
  for j in range(3):
   if len(left)<5:break
   H,k=cv2.estimateAffinePartial2D(p[left].astype('float32'),q[left].astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=8,maxIters=30000,confidence=.999)
   if H is None:break
   ii=left[k.ravel()>0];left=np.setdiff1d(left,ii);fits.append(dict(anchors=len(ii),scale=float(np.hypot(H[0,0],H[1,0])),angle=float(np.degrees(np.arctan2(H[1,0],H[0,0]))),matrix1600=np.vstack([H,[0,0,1]]).tolist(),span=np.ptp(p[ii],axis=0).tolist(),pa=p[ii].tolist(),pb=q[ii].tolist()))
  summary[f'{a}_{b}_{mode}']=dict(matches=len(matches),fits=fits);print(a,b,mode,len(matches),[(f['anchors'],round(f['scale'],2),round(f['angle'],2)) for f in fits],flush=True)
(r/'descriptor_probe.json').write_text(json.dumps(summary,indent=2))
