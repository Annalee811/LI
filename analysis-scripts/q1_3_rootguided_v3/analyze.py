import os
import json,itertools
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw
r=Path(os.environ['LI_Q1_RECHECK_WORK']);meta=json.loads((r/'search_priors.json').read_text());wh={int(k):v for k,v in meta['original_wh'].items()};boxes={int(k):v for k,v in meta['native_boxes'].items()};cv2.setNumThreads(1);cv2.setRNGSeed(20261019)
summary={}
for a,b in [(1,2),(1,3),(2,3)]:
 p=r/f'T{a}_to_T{b}'
 if not(p/'matches.npz').exists():continue
 z=np.load(p/'matches.npz');pa=z['pa']*1600/wh[a][0];pb=z['pb']*1600/wh[b][0];cc=z['confidence'];cy=z['cycle_analysis1600'];models=[]
 print(a,b,'c-quantiles',np.quantile(cc,[0,.5,.8,.9,.95,.99,1]),'cycle',np.quantile(cy,[.1,.5,.9]))
 for ct,er,thr in [(0.05,20,8),(.15,15,8),(.3,10,8),(.5,5,5)]:
  ok=(cc>=ct)&(cy<er);left=np.flatnonzero(ok)
  for j in range(3):
   if len(left)<8:break
   H,k=cv2.estimateAffinePartial2D(pa[left].astype('float32'),pb[left].astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=thr,maxIters=20000,confidence=.999,refineIters=20)
   if H is None:break
   ii=left[k.ravel()>0];left=np.setdiff1d(left,ii);sc=np.hypot(H[0,0],H[1,0]);an=np.degrees(np.arctan2(H[1,0],H[0,0]));err=np.linalg.norm(pa[ii]@H[:,:2].T+H[:,2]-pb[ii],axis=1)
   m=dict(conf_threshold=ct,cycle_limit=er,anchors=len(ii),scale=float(sc),angle_deg=float(an),matrix1600=np.vstack([H,[0,0,1]]).tolist(),source_span=np.ptp(pa[ii],axis=0).tolist(),median_residual=float(np.median(err)),median_conf=float(np.median(cc[ii])),median_cycle=float(np.median(cy[ii])))
   models.append(m);print('fit',m)
   if j==0:
    A=Image.open(r/f'T{a}_search_RGB.png');B=Image.open(r/f'T{b}_search_RGB.png');A.thumbnail((900,600));B.thumbnail((900,600));im=Image.new('RGB',(1800,650),'white');im.paste(A,(0,40));im.paste(B,(900,40));d=ImageDraw.Draw(im)
    rng=np.random.default_rng(0)
    for ix in rng.choice(ii,min(50,len(ii)),replace=False):
     ua=(pa[ix]*wh[a][0]/1600-boxes[a][:2])*A.width/(boxes[a][2]-boxes[a][0])+[0,40];ub=(pb[ix]*wh[b][0]/1600-boxes[b][:2])*B.width/(boxes[b][2]-boxes[b][0])+[900,40];color=tuple(rng.integers(50,220,3).tolist());d.line([tuple(ua),tuple(ub)],fill=color,width=1);d.ellipse([ua[0]-3,ua[1]-3,ua[0]+3,ua[1]+3],fill=color);d.ellipse([ub[0]-3,ub[1]-3,ub[0]+3,ub[1]+3],fill=color)
    d.text((10,10),f'T{a} -> T{b}, confidence>={ct}, {len(ii)} anchors, scale={sc:.3f}, angle={an:.2f}',fill='black');im.save(p/f'fit_ct{ct}.jpg')
 summary[f'T{a}_to_T{b}']=models
(r/'hypotheses.json').write_text(json.dumps(summary,indent=2))
