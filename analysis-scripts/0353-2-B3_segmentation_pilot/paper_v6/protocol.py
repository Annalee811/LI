"""Native-pixel root-only refinement. Immutable v5 masks, unchanged spatial split."""
from pathlib import Path
import csv,hashlib,json,random
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

PERIODS=['T1','T2','T3','T4'];COLS=[22+512*i for i in range(5)];SEED=20261017
SIDE=384;XY=[(x,y) for y in [0,192,384,576,724] for x in [0,64]]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def counts(gt,pred):
 g=gt==255;v=gt!=128;p=pred.astype(bool)
 return np.array([(g&p&v).sum(),(~g&p&v).sum(),(g&~p&v).sum(),(~g&~p&v).sum()],np.int64)
def metrics(cc):
 tp,fp,fn,tn=map(int,cc)
 return dict(TP=tp,FP=fp,FN=fn,TN=tn,Precision=tp/max(1,tp+fp),Recall=tp/max(1,tp+fn),Dice=2*tp/max(1,2*tp+fp+fn),IoU=tp/max(1,tp+fp+fn))
class Data:
 def __init__(self,source):
  self.source=Path(source);self.rgb={};self.gt={};self.roots={}
  provenance=json.loads((self.source/'dataset/provenance.json').read_text())
  for path,digest in provenance['files'].items():assert sha(self.source/path)==digest
  if 'source_progress_sha256' in provenance:assert sha(self.source/'original_human_progress.json')==provenance['source_progress_sha256']
  for t in PERIODS:
   self.rgb[t]=np.array(Image.open(self.source/'dataset/upper'/(t+'_RGB.png')).convert('RGB'))
   self.gt[t]=np.array(Image.open(self.source/'dataset/upper'/(t+'_mask.png')))
   assert self.rgb[t].shape==(1108,2541,3) and self.gt[t].shape==(1108,2541)
   for col,x in enumerate(COLS[:3]):self.roots[t,col]=np.argwhere(self.gt[t][:,x:x+448]==255)
 def sample(self,index):
  t=random.choice(PERIODS);col=random.randrange(3);x0=COLS[col]
  if index%2==0:x,y=random.choice(XY)
  else:
   pool=self.roots[t,col];yy,xx=pool[random.randrange(len(pool))]
   x=int(np.clip(xx-random.randrange(SIDE),0,448-SIDE));y=int(np.clip(yy-random.randrange(SIDE),0,1108-SIDE))
  im=self.rgb[t][y:y+SIDE,x0+x:x0+x+SIDE].copy();gt=self.gt[t][y:y+SIDE,x0+x:x0+x+SIDE].copy()
  k=random.randrange(4);im=np.rot90(im,k);gt=np.rot90(gt,k)
  if random.random()<.5:im=np.fliplr(im);gt=np.fliplr(gt)
  if random.random()<.5:im=np.flipud(im);gt=np.flipud(gt)
  im=im.astype(np.float32)/255;im=np.clip(im*random.uniform(.85,1.15),0,1)**random.uniform(.85,1.15)
  return im.copy(),(gt==255).astype(np.float32),(gt!=128).astype(np.float32)
 def column(self,infer,col,batch=2):
  result={};x0=COLS[col];hh=np.hanning(SIDE).astype(np.float32);blend=np.maximum(hh[:,None]*hh[None,:],.05)
  for t in PERIODS:
   val=np.zeros((1108,448),np.float32);weight=np.zeros_like(val)
   for i in range(0,len(XY),batch):
    coords=XY[i:i+batch];ims=np.stack([self.rgb[t][y:y+SIDE,x0+x:x0+x+SIDE] for x,y in coords]).astype(np.float32)/255
    ps=infer(ims);assert ps.shape==(len(coords),SIDE,SIDE) and np.isfinite(ps).all()
    for (x,y),p in zip(coords,ps):val[y:y+SIDE,x:x+SIDE]+=p*blend;weight[y:y+SIDE,x:x+SIDE]+=blend
   assert (weight>0).all();result[t]=val/weight
  return result
 def evaluate(self,maps,col,threshold):
  cc=np.zeros(4,np.int64);per={};x=COLS[col]
  for t in PERIODS:
   row=counts(self.gt[t][:,x:x+448],maps[t]>=threshold);cc+=row;per[t]=metrics(row)
  return dict(root=metrics(cc),per_period=per,threshold=threshold)
def calibrate(data,maps,col=3):
 rows=[]
 for th in np.arange(.1,.901,.025):
  th=round(float(th),3);rows.append(dict(threshold=th,**data.evaluate(maps,col,th)['root']))
 return max(rows,key=lambda r:(r['Dice'],-abs(r['threshold']-.5))),rows
def save_maps(folder,maps):
 Path(folder).mkdir(parents=True,exist_ok=True)
 for t,p in maps.items():np.save(Path(folder)/(t+'_probability.npy'),p.astype(np.float32))

def clean(prob,cfg,return_removed=False):
 high=cfg['threshold'];low=max(.01,high-cfg.get('hysteresis_delta',0))
 mask=prob>=high
 if low<high:mask=ndi.binary_propagation(mask,structure=np.ones((3,3),bool),mask=prob>=low)
 labels,n=ndi.label(mask,structure=np.ones((3,3),np.uint8));removed=np.zeros_like(mask)
 maxarea=cfg.get('compact_area',0)
 if maxarea and n:
  flat=labels.ravel();area=np.bincount(flat,minlength=n+1).astype(float);yy,xx=np.indices(mask.shape,dtype=np.float32)
  sx=np.bincount(flat,weights=xx.ravel(),minlength=n+1);sy=np.bincount(flat,weights=yy.ravel(),minlength=n+1)
  sxx=np.bincount(flat,weights=(xx*xx).ravel(),minlength=n+1);syy=np.bincount(flat,weights=(yy*yy).ravel(),minlength=n+1)
  sxy=np.bincount(flat,weights=(xx*yy).ravel(),minlength=n+1);aa=np.maximum(area,1)
  vx=np.maximum(0,sxx/aa-(sx/aa)**2);vy=np.maximum(0,syy/aa-(sy/aa)**2);cov=sxy/aa-sx*sy/aa**2
  disc=np.sqrt((vx-vy)**2+4*cov**2);large=(vx+vy+disc)/2;small=np.maximum(0,(vx+vy-disc)/2)
  elong=np.sqrt((large+.25)/(small+.25));maxprob=ndi.maximum(prob,labels,index=np.arange(n+1))
  drop=(area<=maxarea)&(elong<2.5)&(maxprob<min(.99,high+.2));drop[0]=False
  removed=drop[labels];mask=mask&~removed
 return (mask,removed) if return_removed else mask
