"""Two-stage adjacent-period classification; native pixels, immutable user labels."""
from pathlib import Path
import csv, hashlib, json, random
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from junctions import extra_features,semantic_boundary,junction_region

SEED=20261018
PERIODS=['T1','T2','T3','T4']
MODELS=['unet','deeplab','segformer']
COLS=[22+512*i for i in range(5)]
SIDE=256
HEIGHT=1108
PALETTE=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
NAMES={1:'persistent',2:'newly_observed',3:'tip_growth'}
MODEL_NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['phase','draw'],lineterminator='\n');w.writeheader();w.writerows(rows)
def coords(h,w,side=SIDE):
    assert min(h,w)>=side
    xs=list(range(0,w-side+1,side//2));ys=list(range(0,h-side+1,side//2))
    if xs[-1]!=w-side:xs.append(w-side)
    if ys[-1]!=h-side:ys.append(h-side)
    return [(x,y) for y in ys for x in xs]
def target(mask,temporal):
    y=np.zeros(mask.shape,np.uint8)
    y[(mask==128)|((mask==255)&(temporal==0))]=255
    for old,new in [(3,1),(1,2),(2,3)]:y[(mask==255)&(temporal==old)]=new
    return y
def features(previous,current,pp,cp,previous_mask,current_mask):
    """Only RGB and model predictions. Never consumes a human root mask."""
    binary=previous_mask.astype(bool);sk=skeletonize(binary);degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
    cues=[]
    for points in [sk&(degree==1),sk&(degree>=3)]:
        cues.append(np.exp(-ndi.distance_transform_edt(~points)**2/200).astype(np.float32) if points.any() else np.zeros(pp.shape,np.float32))
    distance=np.minimum(ndi.distance_transform_edt(~binary),32).astype(np.float32)/32 if binary.any() else np.ones(pp.shape,np.float32)
    # Both RGB images have predicted background removed before temporal learning.
    base=np.concatenate([previous.astype(np.float32)/255*binary[...,None],current.astype(np.float32)/255*current_mask[...,None],(pp*binary)[...,None],(cp*current_mask)[...,None],cues[0][...,None],cues[1][...,None],distance[...,None]],2)
    return np.concatenate([base,extra_features(binary,current_mask.astype(bool))],2)

class Data:
    def __init__(self,data,roots,model,columns):
        self.path=Path(data);self.roots=Path(roots);self.model=model;self.columns=list(columns)
        self.rgb={t:np.array(Image.open(self.path/'upper'/f'{t}_RGB.png').convert('RGB')) for t in PERIODS}
        self.mask={t:np.array(Image.open(self.path/'upper'/f'{t}_mask.png')) for t in PERIODS}
        self.y={t:np.array(Image.open(self.path/'targets'/f'{t}_temporal_target.png')) for t in PERIODS[1:]}
        self.cache={};self.probs={};self.gates={};self.pools={k:[] for k in [1,2,3]}
        self.bounds={col:(COLS[col],448) for col in self.columns};self.draws=[];self.focus={k:[] for k in ['junction','boundary']}
        self.selection=json.loads((self.roots/model/'postprocessing_selection.json').read_text())
        for col in self.columns:
            left=COLS[col];sl=slice(left,left+448)
            folder=self.roots/model/'probabilities'/('val' if col==3 else 'test/C4' if col==4 else f'train/C{col}')
            for t in PERIODS:self.probs[t,col]=np.load(folder/f'{t}_probability.npy')
            for t in PERIODS:self.gates[t,col]=np.array(Image.open(self.path/'root_gates'/model/f'C{col}'/f'{t}.png'))>0
            for prev,t in zip(PERIODS,PERIODS[1:]):
                self.cache[t,col]=features(self.rgb[prev][:,sl],self.rgb[t][:,sl],self.probs[prev,col],self.probs[t,col],self.gates[prev,col],self.gates[t,col])
                for name,region in [('junction',junction_region(self.gates[t,col])),('boundary',semantic_boundary(self.y[t][:,sl]))]:
                    pool=np.argwhere(region&(self.y[t][:,sl]>0)&(self.y[t][:,sl]<255))
                    if len(pool):self.focus[name].append((t,col,pool))
                for k in [1,2,3]:
                    pool=np.argwhere(self.y[t][:,sl]==k)
                    if len(pool):self.pools[k].append((t,col,pool))
    def add_full(self):
        """All upper pixels, including guards, available only to final deployment fit."""
        self.bounds[5]=(0,self.rgb['T1'].shape[1])
        for t in PERIODS:
            self.probs[t,5]=np.load(self.roots/self.model/'probabilities/upper'/f'{t}_probability.npy')
            self.gates[t,5]=np.array(Image.open(self.roots/self.model/'images'/f'{t}_root_mask.png'))>0
        for prev,t in zip(PERIODS,PERIODS[1:]):
            self.cache[t,5]=features(self.rgb[prev],self.rgb[t],self.probs[prev,5],self.probs[t,5],self.gates[prev,5],self.gates[t,5])
            for name,region in [('junction',junction_region(self.gates[t,5])),('boundary',semantic_boundary(self.y[t]))]:
                pool=np.argwhere(region&(self.y[t]>0)&(self.y[t]<255))
                if len(pool):self.focus[name].append((t,5,pool))
            for k in [1,2,3]:
                pool=np.argwhere(self.y[t]==k)
                if len(pool):self.pools[k].append((t,5,pool))
    def sample(self,index,columns):
        if index%4 in [1,2] and any(r[1] in columns for r in self.focus['junction' if index%4==1 else 'boundary']):
            mode='junction' if index%4==1 else 'boundary';t,col,pool=random.choice([r for r in self.focus[mode] if r[1] in columns]);y,x=pool[random.randrange(len(pool))]
            x=int(np.clip(x-random.randrange(64,192),0,self.bounds[col][1]-SIDE));y=int(np.clip(y-random.randrange(64,192),0,HEIGHT-SIDE))
        elif index%4<3:
            k=1+index%3;available=[r for r in self.pools[k] if r[1] in columns]
            t,col,pool=random.choice(available);y,x=pool[random.randrange(len(pool))]
            x=int(np.clip(x-random.randrange(SIDE),0,self.bounds[col][1]-SIDE));y=int(np.clip(y-random.randrange(SIDE),0,HEIGHT-SIDE))
        else:
            t=random.choice(PERIODS[1:]);col=random.choice(columns);x=random.randrange(self.bounds[col][1]-SIDE+1);y=random.randrange(HEIGHT-SIDE+1)
        im=self.cache[t,col][y:y+SIDE,x:x+SIDE].copy();left=self.bounds[col][0]+x;gt=self.y[t][y:y+SIDE,left:left+SIDE].copy()
        rotation=random.randrange(4);im=np.rot90(im,rotation);gt=np.rot90(gt,rotation)
        fx=random.random()<.5;fy=random.random()<.5
        if fx:im=np.fliplr(im);gt=np.fliplr(gt)
        if fy:im=np.flipud(im);gt=np.flipud(gt)
        brightness=[random.uniform(.95,1.05),random.uniform(.95,1.05)]
        for sl,factor in zip([slice(0,3),slice(3,6)],brightness):im[:,:,sl]=np.clip(im[:,:,sl]*factor,0,1)
        self.draws.append(dict(phase='final_all_upper' if col==5 else 'evaluation_fit',draw=len(self.draws),current=t,column=col,x=left,y=y,side=SIDE,rotation_quarters=rotation,flipx=fx,flipy=fy,previous_brightness=brightness[0],current_brightness=brightness[1]))
        return im.transpose(2,0,1).copy(),gt.copy()

def predict(model,feature,device,batch=4):
    import torch
    h,w=feature.shape[:2];xy=coords(h,w);blend=np.maximum(np.outer(np.hanning(SIDE),np.hanning(SIDE)),.05).astype(np.float32)
    total=np.zeros((3,h,w),np.float32);weight=np.zeros((h,w),np.float32);model.eval()
    with torch.inference_mode():
        for i in range(0,len(xy),batch):
            cc=xy[i:i+batch];ims=np.stack([feature[y:y+SIDE,x:x+SIDE].transpose(2,0,1) for x,y in cc])
            xx=torch.from_numpy(ims.copy()).to(device).contiguous(memory_format=torch.channels_last)
            enabled=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=enabled):logits=model(xx)
            p=logits.float().softmax(1).cpu().numpy()
            for (x,y),v in zip(cc,p):total[:,y:y+SIDE,x:x+SIDE]+=v*blend;weight[y:y+SIDE,x:x+SIDE]+=blend
    assert (weight>0).all()
    result=total/weight[None]
    if hasattr(model,'class_multipliers'):
        # Declared probability quantization makes cached and fresh inference agree.
        result=result.astype(np.float16).astype(np.float32)*np.asarray(model.class_multipliers,np.float32)[:,None,None]
        result/=np.maximum(result.sum(0,keepdims=True),1e-12)
    if hasattr(model,'class_adjustment'):
        result*=np.asarray(model.class_adjustment,np.float32)[:,None,None]
        result/=np.maximum(result.sum(0,keepdims=True),1e-12)
    return result

def confusion(gt,pred,root_only=False):
    valid=(gt!=255)&((gt>0) if root_only else True)
    g=gt[valid].astype(np.int64);p=pred[valid].astype(np.int64)
    return np.bincount(g*4+p,minlength=16).reshape(4,4)
def metrics(cm):
    rows=[]
    for k in [1,2,3]:
        tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp);support=int(cm[k].sum())
        rows.append(dict(class_id=k,name=NAMES[k],support_pixels=support,TP=tp,FP=fp,FN=fn,
            Precision=tp/(tp+fp) if tp+fp else None,Recall=tp/support if support else None,
            Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,IoU=tp/(tp+fp+fn) if tp+fp+fn else None))
    return dict(classes=rows,macro_Dice=float(np.mean([r['Dice'] for r in rows if r['Dice'] is not None])) if any(r['Dice'] is not None for r in rows) else None,confusion_matrix=cm.tolist())
