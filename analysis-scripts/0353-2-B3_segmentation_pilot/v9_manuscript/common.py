"""Strict v9 re-analysis; no model fitting, label changes or new thresholds."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
MODELS=['unet','deeplab','segformer'];NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
PERIODS=['T1','T2','T3','T4'];COLS=[22+512*i for i in range(5)];METRICS=['Precision','Recall','Dice','IoU'];COLORS=['#0072B2','#D55E00','#009E73'];ORIGINAL_COMMIT='608659941cd720a8292efbebda782e5277e30f08'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def js(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(path,rows):
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvread(path):
 with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def counts(gt,pred):
 g=gt==255;v=gt!=128;p=pred>0
 return np.array([(g&p&v).sum(),(~g&p&v).sum(),(g&~p&v).sum(),(~g&~p&v).sum()],np.int64)
def metrics(cc):
 tp,fp,fn,tn=map(int,cc)
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,Precision=tp/(tp+fp) if tp+fp else None,Recall=tp/(tp+fn) if tp+fn else None,Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,IoU=tp/(tp+fp+fn) if tp+fp+fn else None)
def length_field(mask):
 sk=skeletonize(mask);out=np.zeros(sk.shape,np.float64)
 for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
  y1=slice(0,-dy) if dy else slice(None);y2=slice(dy,None);x1=slice(0,-dx) if dx>0 else slice(-dx,None) if dx<0 else slice(None);x2=slice(dx,None) if dx>0 else slice(0,dx) if dx<0 else slice(None)
  edges=sk[y1,x1]&sk[y2,x2]
  if dx and dy:edges&=~(sk[y1,x2]|sk[y2,x1])
  val=edges*.5*math.hypot(dy,dx);out[y1,x1]+=val;out[y2,x2]+=val
 out[sk&(ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')==1)]=1
 return out
def agreement(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);d=y-x;sst=((x-x.mean())**2).sum();den=x.var()+y.var()+(x.mean()-y.mean())**2;sd=d.std(ddof=1)
 return dict(n_correlated_block_date_pairs=len(x),R2_agreement=float(1-(d*d).sum()/sst) if sst else None,RMSE_px=float(np.sqrt((d*d).mean())),MAE_px=float(abs(d).mean()),bias_px=float(d.mean()),CCC=float(2*((x-x.mean())*(y-y.mean())).mean()/den) if den else None,LoA_lower_px=float(d.mean()-1.96*sd),LoA_upper_px=float(d.mean()+1.96*sd))
