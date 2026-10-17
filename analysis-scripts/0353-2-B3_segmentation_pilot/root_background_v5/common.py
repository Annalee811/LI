"""Shared, Python 3.5-compatible native-pixel binary segmentation protocol."""
from __future__ import division, print_function
import csv, hashlib, json, random
from pathlib import Path
import numpy as np
from PIL import Image

SEED = 20261017
SIDE = 256
PERIODS = ['T1', 'T2', 'T3', 'T4']
COLS = [22 + 512*i for i in range(5)]
COORDS = [(x,y) for y in [0,256,512,768,852] for x in [0,192]]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def counts(gt, pred):
    g=gt==255; p=pred.astype(bool); v=gt!=128
    return np.array([(g&p&v).sum(),(~g&p&v).sum(),(g&~p&v).sum(),(~g&~p&v).sum()],np.int64)
def metrics(cc):
    tp,fp,fn,tn=map(int,cc)
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,Precision=tp/max(1,tp+fp),Recall=tp/max(1,tp+fn),
                Dice=2*tp/max(1,2*tp+fp+fn),IoU=tp/max(1,tp+fp+fn))

class Data:
    def __init__(self, root):
        self.root=Path(root); self.rgb={}; self.gt={}; self.rootcoords={}
        for t in PERIODS:
            self.rgb[t]=np.array(Image.open(str(self.root/'dataset/upper'/(t+'_RGB.png'))).convert('RGB'))
            self.gt[t]=np.array(Image.open(str(self.root/'dataset/upper'/(t+'_mask.png'))))
            assert self.rgb[t].shape==(1108,2541,3) and self.gt[t].shape==(1108,2541)
            for col,x in enumerate(COLS[:3]):
                self.rootcoords[t,col]=np.argwhere(self.gt[t][:,x:x+448]==255)
    def sample(self, index, augment=True):
        t=random.choice(PERIODS); col=random.randrange(3); x0=COLS[col]
        if index%2==0:
            dx,y=random.choice(COORDS)
        else:
            coords=self.rootcoords[t,col]; yy,xx=coords[random.randrange(len(coords))]
            dx=int(np.clip(xx-random.randrange(SIDE),0,448-SIDE))
            y=int(np.clip(yy-random.randrange(SIDE),0,1108-SIDE))
        im=self.rgb[t][y:y+SIDE,x0+dx:x0+dx+SIDE].copy()
        gt=self.gt[t][y:y+SIDE,x0+dx:x0+dx+SIDE].copy()
        if augment:
            k=random.randrange(4); im=np.rot90(im,k); gt=np.rot90(gt,k)
            if random.random()<.5: im=np.fliplr(im); gt=np.fliplr(gt)
            if random.random()<.5: im=np.flipud(im); gt=np.flipud(gt)
        im=im.astype(np.float32)/255
        if augment: im=np.clip(im*random.uniform(.9,1.1),0,1)**random.uniform(.9,1.1)
        return im.copy(),(gt==255).astype(np.float32),(gt!=128).astype(np.float32)
    def column(self, infer, col, batch=4):
        result={}; x0=COLS[col]
        for t in PERIODS:
            value=np.zeros((1108,448),np.float32); weight=np.zeros_like(value)
            for i in range(0,len(COORDS),batch):
                coords=COORDS[i:i+batch]
                ims=np.stack([self.rgb[t][y:y+SIDE,x0+x:x0+x+SIDE] for x,y in coords]).astype(np.float32)/255
                probs=infer(ims)
                assert probs.shape==(len(coords),SIDE,SIDE) and np.isfinite(probs).all()
                for (x,y),p in zip(coords,probs): value[y:y+SIDE,x:x+SIDE]+=p; weight[y:y+SIDE,x:x+SIDE]+=1
            assert (weight>0).all()
            result[t]=value/weight
        return result
    def evaluate(self, maps, col, threshold):
        total=np.zeros(4,np.int64); per={}; x=COLS[col]
        for t in PERIODS:
            cc=counts(self.gt[t][:,x:x+448],maps[t]>=threshold);total+=cc;per[t]=metrics(cc)
        return dict(root=metrics(total),per_period=per,n_unique_pixels=4*1108*448,threshold=threshold)

def calibrate(data, maps, col):
    rows=[]
    for th in np.arange(.1,.901,.025):
        th=float(round(float(th),3)); rows.append(dict(threshold=th,**data.evaluate(maps,col,th)['root']))
    chosen=max(rows,key=lambda r:(r['Dice'],-abs(r['threshold']-.5)))
    return chosen['threshold'],rows

def save_maps(folder,maps):
    Path(folder).mkdir(parents=True,exist_ok=True)
    for t,p in maps.items(): np.save(str(Path(folder)/(t+'_probability.npy')),p.astype(np.float32))
