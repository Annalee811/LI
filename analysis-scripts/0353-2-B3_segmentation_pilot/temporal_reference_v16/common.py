"""U-Net-derived temporal reference; preserved original human labels are an audit."""
import csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
HERE=Path(__file__).resolve().parent;V13=HERE.parent/'temporal_v13';sys.path.insert(0,str(V13))
spec=importlib.util.spec_from_file_location('v13_common',V13/'common.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
from model import TemporalHead,loss
import axes
PERIODS=['T1','T2','T3','T4'];MODELS=['unet','deeplab','segformer'];MODEL_NAMES=old.MODEL_NAMES;NAMES=old.NAMES;PALETTE=old.PALETTE;SEED=20261021;SIDE=256

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def js(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(path):
 with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]) if data else ['empty'],lineterminator='\n');w.writeheader();w.writerows(data)
def unpack(v,shape):return np.repeat(np.array(v[::2],np.uint8),np.array(v[1::2],np.int64)).reshape(shape)
def pack(a):
 a=a.ravel();s=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1];e=np.r_[s[1:],len(a)];return np.stack([a[s].astype(np.int64),e-s],1).ravel().tolist()
class AxisHead(TemporalHead):
 def __init__(self):super().__init__();self.a[0]=nn.Conv2d(31,8,3,padding=1,bias=False)
def load_head(path,device,warm=False):
 ck=torch.load(path,map_location='cpu',weights_only=True);model=AxisHead();state=ck['state_dict'].copy();oldweight=state['a.0.weight']
 if oldweight.shape[1]!=31:
  z=torch.zeros_like(model.a[0].weight);z[:,:oldweight.shape[1]]=oldweight;state['a.0.weight']=z
 model.load_state_dict(state);model.class_multipliers=[1.,1.,1.] if warm else ck.get('class_multipliers',[1.,1.,1.]);model.class_adjustment=[1.,1.,1.] if warm else ck.get('class_adjustment',[1.,1.,1.]);return model.to(device,memory_format=torch.channels_last)
def prob(model,x,amp=False):
 with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=amp):z=model(x)
 return z.float().softmax(1)
def predict_probability(model,feature,device,batch=4):
 h,w=feature.shape[:2];coords=old.coords(h,w);blend=np.maximum(np.outer(np.hanning(256),np.hanning(256)),.05).astype('float32');total=np.zeros((3,h,w),np.float32);weight=np.zeros((h,w),np.float32);model.eval();amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
 with torch.inference_mode():
  for i in range(0,len(coords),batch):
   cc=coords[i:i+batch];xx=np.stack([feature[y:y+256,x:x+256].transpose(2,0,1) for x,y in cc]).astype('float32');q=prob(model,torch.from_numpy(xx).to(device).contiguous(memory_format=torch.channels_last),amp).cpu().numpy()
   for (x,y),value in zip(cc,q):total[:,y:y+256,x:x+256]+=value*blend;weight[y:y+256,x:x+256]+=blend
 assert (weight>0).all();return total/weight[None]
def load_axes(path):
 z=np.load(path);meta=json.loads(Path(path).with_suffix('.json').read_text());return dict(xy=z['xy'],paths=[z[f'path_{i}'] for i in range(len(meta['endkinds']))],endkinds=meta['endkinds'],axis_map=z['axis_map'],uncertain=z['uncertain'],decisions=meta['decisions'])
class Data:
 def __init__(self,run,roots,model):
  self.run=Path(run);self.roots=Path(roots);self.model=model;self.pairs=rows(self.run/'data/paired_manifest.csv');self.train=[x for x in self.pairs if x['split']=='train'];self.rgb={t:np.array(Image.open(self.run/'data/upper'/f'{t}_RGB.png')) for t in PERIODS};self.gate={t:np.array(Image.open(self.roots/model/'upper'/f'{t}_root_mask.png'))>0 for t in PERIODS};self.rootprob={t:np.load(self.roots/model/'upper'/f'{t}_probability.npy') for t in PERIODS};self.targets={ref:{t:np.array(Image.open(self.run/'data'/ref/f'{t}_classes.png')) for t in PERIODS} for ref in ['original','revised']};self.features={t:np.load(self.run/'feature_cache'/model/(t+'.npy'),mmap_mode='r') for t in PERIODS[1:]};self.axes={t:load_axes(self.run/'axis_cache'/model/(t+'.npz')) for t in PERIODS};self.links={t:np.load(self.run/'axis_cache'/model/(t+'_matches.npz')) for t in PERIODS[1:]}
  self.link={t:dict(axis_info=json.loads((self.run/'axis_cache'/model/(t+'.json')).read_text())['axis_matches'],node_score=self.links[t]['node_score'],nearest_previous_axis=self.links[t]['nearest_previous_axis'],nearest_distance=self.links[t]['nearest_distance']) for t in PERIODS[1:]}
 def patch(self,row,train=False,rng=None):
  x,y=int(row['x']),int(row['y']);t=row['current'];feat=self.features[t][y:y+SIDE,x:x+SIDE].astype('float32');gt=self.targets['revised'][t][y:y+SIDE,x:x+SIDE].copy();aug={}
  if train:
   gt[~self.gate[t][y:y+SIDE,x:x+SIDE]]=255;k=int(rng.integers(4));fx=bool(rng.integers(2));fy=bool(rng.integers(2));feat=np.rot90(feat,k);gt=np.rot90(gt,k)
   if fx:feat=np.fliplr(feat);gt=np.fliplr(gt)
   if fy:feat=np.flipud(feat);gt=np.flipud(gt)
   brightness=[float(rng.uniform(.95,1.05)),float(rng.uniform(.95,1.05))]
   for sl,b in zip([slice(0,3),slice(3,6)],brightness):feat[:,:,sl]=np.clip(feat[:,:,sl]*b,0,1)
   aug=dict(rotation_quarters=k,flip_x=fx,flip_y=fy,previous_brightness=brightness[0],current_brightness=brightness[1])
  return feat.transpose(2,0,1).copy(),gt.copy(),aug
 def counts(self):
  cc=np.zeros(3,np.int64)
  for row in self.train:
   _,gt,_=self.patch(row);x,y=int(row['x']),int(row['y']);gate=self.gate[row['current']][y:y+256,x:x+256];cc+=[int(((gt==k)&gate).sum()) for k in [1,2,3]]
  return cc

def evaluate_tiles(model,data,split,device,amp=False):
 cms={ref:np.zeros((4,4),np.int64) for ref in ['original','revised']};selected=[r for r in data.pairs if r['split']==split];model.eval()
 with torch.inference_mode():
  for i in range(0,len(selected),4):
   rr=selected[i:i+4];xx=np.stack([data.patch(r)[0] for r in rr]);pp=prob(model,torch.from_numpy(xx).to(device).contiguous(memory_format=torch.channels_last),amp).cpu().numpy()
   for row,q in zip(rr,pp):
    x,y=int(row['x']),int(row['y']);t=row['current'];pred=np.where(data.gate[t][y:y+256,x:x+256],q.argmax(0)+1,0).astype('uint8')
    for ref in cms:cms[ref]+=old.confusion(data.targets[ref][t][y:y+256,x:x+256],pred)
 return {ref:old.metrics(cm) for ref,cm in cms.items()}

# Keep this version's decoder/predictor ahead of legacy v13 helpers.
sys.path.insert(0,str(HERE))
