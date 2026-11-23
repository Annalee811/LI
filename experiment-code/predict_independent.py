from pathlib import Path
import json,csv,time,hashlib
import numpy as np,torch
from scipy import ndimage as ndi
from PIL import Image
from study_model import Network
W=Path(__file__).resolve().parents[1];OLD=Path('/workspace/work/V90_all_microtubes_20261118');OUT=W/'independent_predictions';cache=W/'external_inputs';cache.mkdir(exist_ok=True)
def cue(mask):
 from skimage.morphology import skeletonize
 sk=skeletonize(mask);degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
 def f(p):return np.exp(-ndi.distance_transform_edt(~p)**2/128).astype(np.float32)if p.any()else np.zeros(mask.shape,np.float32)
 return f(sk&(degree==1)),f(sk&(degree>=3))
def prepare(pos):
 C=cache/pos;C.mkdir(exist_ok=True);src=OLD/'transfer_candidates'/pos;meta=json.loads((src/'adapter_provenance.json').read_text());g=dict(np.load(src/'local/data/unet/graph.npz'));owners=dict(np.load(src/'local/data/unet/ownership.npz'));prev=None
 for t in meta['periods']:
  rgb=np.array(Image.open(src/f'base/data/upper/{t}_RGB.png')).astype(np.float32)/255;mask=np.array(Image.open(src/f'base/unet/full/{t}_classes.png'))>0;end,jun=cue(mask)
  if not(C/(t+'.npz')).exists():
   y,x=np.where(mask);node=owners[t][y,x];c=g['centers'][node];p=np.load(OLD/'inputs_t045'/pos/(t+'_prior_probability.npz'))['probability'].astype(np.float32);delta=np.zeros_like(rgb)if prev is None else rgb-prev[0];distance=np.zeros(mask.shape,np.float32)if prev is None else ndi.distance_transform_edt(~prev[1]).astype(np.float32)/32;pend=end if prev is None else prev[2];pf=np.c_[np.log(p[:,y,x].T.clip(1e-6)),(x-c[:,0])/12,(y-c[:,1])/12,rgb[y,x],delta[y,x],np.minimum(distance[y,x],1),pend[y,x],end[y,x],jun[y,x]].astype(np.float32);np.savez_compressed(C/(t+'.npz'),pixel=y*mask.shape[1]+x,node=node,pf=pf,shape=np.array(mask.shape));print('EXTERNAL_INPUT',pos,t,len(x),flush=True)
  prev=(rgb,mask,end)
 return src,meta,g,C

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);rows=list(csv.DictReader((W/'reports/annotation_checklist.csv').open(encoding='utf-8-sig')));positions=list(dict.fromkeys(r['position_id']for r in rows));data={p:prepare(p)for p in positions};norm=dict(np.load(W/'data/train_only_normalization.npz'));protocol=json.loads((W/'reports/locked_protocol.json').read_text());jobs=[(v,s)for s in protocol['seeds']for v in protocol['models']];done=[]
 for variant,seed in jobs:
  ckpath=W/'experiments'/variant/str(seed)/'best.pt';marker=ckpath.parent/'status.json'
  for _ in range(720):
   if marker.exists()and json.loads(marker.read_text()).get('stage')=='complete':break
   time.sleep(5)
  else:raise RuntimeError('Model not completed: '+str(ckpath))
  ck=torch.load(ckpath,map_location='cpu',weights_only=False);net=Network(variant);net.load_state_dict(ck['backbone']);net.eval();cksha=hashlib.sha256(ckpath.read_bytes()).hexdigest()
  for pos in positions:
   dest=OUT/variant/str(seed)/pos;dest.mkdir(parents=True,exist_ok=True);mark=dest/'inference_status.json'
   if mark.exists()and json.loads(mark.read_text()).get('checkpoint_SHA256')==cksha:continue
   src,meta,g,C=data[pos];extra=np.load(src/'local/data/unet/extra.npz')['geometry'];frames=np.load(src/'local/data/unet/frames.npz')['frames'].astype(np.float32);d={k:torch.from_numpy(g[k])for k in ['pairs','attrs','phase','centers']};d.update(x=torch.from_numpy(((g['x']-norm['x_mean'])/norm['x_std']).astype(np.float32)),extra=torch.from_numpy(((extra-norm['geometry_mean'])/norm['geometry_std']).astype(np.float32)),frames=torch.from_numpy(frames),part=torch.zeros(len(g['x']),dtype=torch.int64),partition_frame_masks=torch.ones((1,*frames.shape[-2:])),native_width=meta['native_width'],native_height=meta['native_height']);st=time.monotonic()
   with torch.inference_mode():h,_,_=net(d,external=True)
   for ti,t in enumerate(meta['periods']):
    q=dict(np.load(C/(t+'.npz')));image=np.zeros(tuple(q['shape']),np.uint8)
    if ti==0:image.ravel()[q['pixel']]=4
    else:
     with torch.inference_mode():
      for i in range(0,len(q['pixel']),32768):
       sl=slice(i,i+32768);p=net.pixel_logits(h,torch.from_numpy(q['node'][sl]),torch.from_numpy(q['pf'][sl])).argmax(1).numpy()+1;image.ravel()[q['pixel'][sl]]=p.astype(np.uint8)
    Image.fromarray(image).save(dest/(t+'_classes.png'))
   mark.write_text(json.dumps(dict(status='prediction_complete_GT_pending',variant=variant,seed=seed,checkpoint_SHA256=cksha,position=pos,periods=meta['periods'],GT_read_at_inference=False,accuracy=None,adapter='Fixed 2026-11-18 transfer graph adapter, with known approximation; shared across all models',seconds=time.monotonic()-st),indent=2));print('INDEPENDENT_PREDICTION',variant,seed,pos,round(time.monotonic()-st,1),flush=True)
  done.append(dict(variant=variant,seed=seed,status='predicted_no_GT_scores'));(W/'reports/independent_prediction_progress.json').write_text(json.dumps(done,indent=2))
 (W/'INDEPENDENT_PREDICTIONS_COMPLETE.json').write_text(json.dumps(dict(models=len(jobs),positions=len(positions),frames_per_model=len(rows),GT_available=False,metrics=None),indent=2));print('ALL_EXTERNAL_PREDICTIONS_COMPLETE',len(jobs)*len(rows),flush=True)
if __name__=='__main__':main()
