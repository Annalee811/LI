"""Versioned U-Net temporal distillation references and paired random tiles."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;V13=HERE.parent/'temporal_v13';sys.path.insert(0,str(V13))
spec=importlib.util.spec_from_file_location('temporal13_common',V13/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from model import TemporalHead,warm_start,loss
from coherence import decode,fragmentation
PERIODS=['T1','T2','T3','T4'];MODELS=['unet','deeplab','segformer'];NAMES=old.NAMES;PALETTE=old.PALETTE;MODEL_NAMES=old.MODEL_NAMES;SEED=20261020;SIDE=256

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def js(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(path):
 with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(data[0]) if data else ['phase','draw']);w.writeheader();w.writerows(data)
def unpack(v,shape):return np.repeat(np.asarray(v[::2],np.uint8),np.asarray(v[1::2],np.int64)).reshape(shape)
def pack(a):
 a=a.ravel();s=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1];e=np.r_[s[1:],len(a)];return np.stack([a[s].astype('int64'),e-s],1).ravel().tolist()
def original_target(mask,temporal,baseline=False):
 if baseline:return np.where(mask==255,4,0).astype('uint8')
 return old.target(mask,temporal)
def load_head(path,device):
 m=warm_start(path,device);ck=__import__('torch').load(path,map_location='cpu',weights_only=True);m.chain_penalty=float(ck.get('chain_penalty',0));return m

def probabilities(model,x,amp=False):
 import torch
 with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=amp):logit=model(x)
 multiplier=np.asarray(getattr(model,'class_multipliers',[1.,1.,1.]))*np.asarray(getattr(model,'class_adjustment',[1.,1.,1.]))
 return (logit.float()+torch.tensor(np.log(multiplier),device=x.device,dtype=torch.float32)[None,:,None,None]).softmax(1)

class Data:
 def __init__(self,run,roots,name,confidence_min=.5):
  self.run=Path(run);self.roots=Path(roots);self.name=name;self.confidence_min=confidence_min;self.pairs=rows(self.run/'data/paired_manifest.csv')
  self.rgb={t:np.array(Image.open(self.run/'data/upper'/f'{t}_RGB.png').convert('RGB')) for t in PERIODS}
  self.gate={t:np.array(Image.open(self.roots/name/'upper'/f'{t}_root_mask.png'))>0 for t in PERIODS}
  self.prob={t:np.load(self.roots/name/'upper'/f'{t}_probability.npy') for t in PERIODS}
  self.target={ref:{t:np.array(Image.open(self.run/'data'/ref/f'{t}_classes.png')) for t in PERIODS} for ref in ['original','revised']}
  self.conf={t:np.array(Image.open(self.run/'data/revised'/f'{t}_confidence_uint16.png')).astype('float32')/65535 for t in PERIODS}
  self.features={};cache=self.run/'feature_cache'/name;cache.mkdir(parents=True,exist_ok=True)
  for prev,t in zip(PERIODS,PERIODS[1:]):
   file=cache/(t+'.npy')
   pins=dict(previous_RGB=sha(self.run/'data/upper'/f'{prev}_RGB.png'),current_RGB=sha(self.run/'data/upper'/f'{t}_RGB.png'),previous_probability=sha(self.roots/name/'upper'/f'{prev}_probability.npy'),current_probability=sha(self.roots/name/'upper'/f'{t}_probability.npy'),previous_gate=sha(self.roots/name/'upper'/f'{prev}_root_mask.png'),current_gate=sha(self.roots/name/'upper'/f'{t}_root_mask.png'),feature_source_sha256=sha(V13/'common.py'),geometry_source_sha256=sha(V13/'junctions.py'))
   meta=cache/(t+'_provenance.json')
   if file.exists():assert json.loads(meta.read_text())==pins
   else:
    value=old.features(self.rgb[prev],self.rgb[t],self.prob[prev],self.prob[t],self.gate[prev],self.gate[t]);assert value.shape[2]==25 and np.isfinite(value).all();np.save(file,value);js(meta,pins);del value
   self.features[t]=np.load(file,mmap_mode='r')
  self.train=[r for r in self.pairs if r['split']=='train']
 def patch(self,row,train=False,rng=None):
  x,y=int(row['x']),int(row['y']);t=row['current'];sl=np.s_[y:y+SIDE,x:x+SIDE]
  feat=self.features[t][sl].copy();gt=self.target['revised'][t][sl].copy()
  if train:
   gt[(self.conf[t][sl]<self.confidence_min)|~self.gate[t][sl]]=255
   k=int(rng.integers(4));fx=bool(rng.integers(2));fy=bool(rng.integers(2));feat=np.rot90(feat,k);gt=np.rot90(gt,k)
   if fx:feat=np.fliplr(feat);gt=np.fliplr(gt)
   if fy:feat=np.flipud(feat);gt=np.flipud(gt)
   brightness=[float(rng.uniform(.95,1.05)),float(rng.uniform(.95,1.05))]
   for ss,factor in zip([slice(0,3),slice(3,6)],brightness):feat[:,:,ss]=np.clip(feat[:,:,ss]*factor,0,1)
   aug=dict(rotation_quarters=k,flip_x=fx,flip_y=fy,previous_brightness=brightness[0],current_brightness=brightness[1])
  else:aug={}
  return feat.transpose(2,0,1).copy(),gt.copy(),aug
 def class_counts(self):
  count=np.zeros(3,np.int64)
  for row in self.train:
   x,y=int(row['x']),int(row['y']);t=row['current'];sl=np.s_[y:y+SIDE,x:x+SIDE];yy=self.target['revised'][t][sl];valid=(self.conf[t][sl]>=self.confidence_min)&self.gate[t][sl]
   count+=np.array([((yy==k)&valid).sum() for k in [1,2,3]])
  return count

def evaluate(model,data,split,device,out=None,batch=4,amp=False):
 import torch
 selected=[r for r in data.pairs if r['split']==split];cms={ref:np.zeros((4,4),np.int64) for ref in ['original','revised']};conds={ref:np.zeros((4,4),np.int64) for ref in cms};per=[]
 model.eval()
 if out:Path(out).mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for i in range(0,len(selected),batch):
   rr=selected[i:i+batch];xx=np.stack([data.patch(r)[0] for r in rr]);pr=probabilities(model,torch.from_numpy(xx).to(device).contiguous(memory_format=torch.channels_last),amp).cpu().numpy()
   for row,p in zip(rr,pr):
    x,y=int(row['x']),int(row['y']);t=row['current'];sl=np.s_[y:y+SIDE,x:x+SIDE];gate=data.gate[t][sl];pred=decode(p,gate,model.chain_penalty)
    if out:
     Image.fromarray(pred).save(Path(out)/(row['pair_id']+'_prediction.png'));np.savez_compressed(Path(out)/(row['pair_id']+'_probability.npz'),probability=p.astype('float16'))
    for ref in cms:
     gt=data.target[ref][t][sl];cm=old.confusion(gt,pred);cms[ref]+=cm;cg=gt.copy();cg[~gate]=255;conds[ref]+=old.confusion(cg,pred,True)
     for v in old.metrics(cm)['classes']:per.append(dict(pair_id=row['pair_id'],current=t,split=split,reference=ref,**v))
 reports={ref:dict(end_to_end=old.metrics(cms[ref]),conditional_on_detected_reference_roots=old.metrics(conds[ref]),pair_count=len(selected)) for ref in cms}
 if out:js(Path(out)/'metrics.json',reports);csvout(Path(out)/'per_pair_class_metrics.csv',per)
 return reports
