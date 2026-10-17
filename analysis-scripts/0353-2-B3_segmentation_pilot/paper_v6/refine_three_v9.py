#!/usr/bin/env python3
"""Three-model continuity fine-tuning; untouched annotations and spatial groups."""
import argparse,json,random,sys,time,shutil
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import protocol as p
from train import RootUNet,inference,skel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
import common as c
from train_torch import create

def objective(logit,y,v):
 b=F.binary_cross_entropy_with_logits(logit,y,reduction='none');g=y*v;q=logit.sigmoid()*v
 thin=(F.avg_pool2d(g,5,1,2)<.3)&(g>.5);weight=1+thin.float()
 pos=(g>.5)&(v>.5);neg=(g<.5)&(v>.5);pv=b[pos]*weight[pos];nv=b[neg]
 hard=nv.topk(max(1,int(nv.numel()*.2))).values
 bce=(pv.sum()+hard.sum())/(weight[pos].sum()+len(hard)).clamp_min(1)
 dice=1-(2*(q*g).sum()+1)/(q.sum()+g.sum()+1)
 sp=skel(q);sg=skel(g);pr=((sp*g).sum()+1)/(sp.sum()+1);re=((sg*q).sum()+1)/(sg.sum()+1)
 return bce+dice+.4*(1-2*pr*re/(pr+re+1e-7))

def full_maps(data,infer,side):
 han=np.hanning(side).astype(np.float32);blend=np.maximum(han[:,None]*han[None,:],.05)
 result={}
 for t,rgb in data.rgb.items():
  h,w=rgb.shape[:2];xs=list(range(0,w-side+1,side//2));ys=list(range(0,h-side+1,side//2))
  if xs[-1]!=w-side:xs.append(w-side)
  if ys[-1]!=h-side:ys.append(h-side)
  coords=[(x,y) for y in ys for x in xs];s=np.zeros((h,w),np.float32);weight=np.zeros_like(s)
  for i in range(0,len(coords),2):
   xy=coords[i:i+2];ims=np.stack([rgb[y:y+side,x:x+side] for x,y in xy]).astype(np.float32)/255
   for (x,y),prob in zip(xy,infer(ims)):s[y:y+side,x:x+side]+=prob*blend;weight[y:y+side,x:x+side]+=blend
  assert (weight>0).all();result[t]=s/weight
 return result

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--model',choices=['unet','segformer'],required=True);a.add_argument('--initial',type=Path,required=True);a.add_argument('--epochs',type=int,default=8);a.add_argument('--infer-only',action='store_true');args=a.parse_args()
 torch.set_num_threads(3);torch.manual_seed(p.SEED);np.random.seed(p.SEED);random.seed(p.SEED)
 validated=p.Data(args.data);train=c.Data(args.data);val=validated if args.model=='unet' else train
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=(RootUNet() if args.model=='unet' else create('segformer')).to(device)
 out=args.run/args.model;out.mkdir(parents=True,exist_ok=True)
 model.load_state_dict(torch.load(out/'best.pt' if args.infer_only else args.initial,map_location=device,weights_only=True)['state_dict'])
 infer=inference(model,device,args.model=='unet');start=time.monotonic()
 if not args.infer_only:
  config=dict(model=args.model,initial_checkpoint_SHA256=p.sha(args.initial),epochs=args.epochs,samples_per_epoch=32,batch=2,training_patch_size=256,
   inference_patch_size=384 if args.model=='unet' else 256,TTA='four flips' if args.model=='unet' else 'none',seed=p.SEED,CUDA=torch.cuda.is_available(),
   objective='Thin-labelled-root weighted BCE (2x, local5x5 occupancy<0.3), hardest20% background +soft Dice +0.4clDice',
   annotations_edited=False,train_columns=[0,1,2],validation_column=3,external_test_images=0,C4_status='Repeated same-position benchmark; excluded from fitting and selection',
   validation_selection='Maximum calibrated strict C3 Dice among epoch0,2,4,6,8; original start can be retained',RGB_only=True,death=False)
  p.js(out/'config.json',config);opt=torch.optim.AdamW(model.parameters(),lr=1e-5 if args.model=='unet' else 5e-5,weight_decay=1e-4)
  maps=val.column(infer,3,2);row,grid=p.calibrate(val,maps);best=row['Dice'];hist=[dict(epoch=0,loss=None,validation_Dice=best,threshold=row['threshold'])]
  shutil.copyfile(args.initial,out/'best.pt');p.save_maps(out/'probabilities/val',maps);p.js(out/'selection.json',dict(epoch=0,**row));p.csvout(out/'thresholds.csv',grid)
  for epoch in range(1,args.epochs+1):
   model.train()
   # Frozen BN moving statistics for both architectures; affine weights train.
   for mod in model.modules():
    if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.eval()
   losses=[]
   for i in range(0,32,2):
    ss=[train.sample(i+j) for j in range(2)];x=torch.from_numpy(np.stack([s[0] for s in ss]).transpose(0,3,1,2)).to(device);y=torch.from_numpy(np.stack([s[1] for s in ss])[:,None]).to(device);v=torch.from_numpy(np.stack([s[2] for s in ss])[:,None]).to(device)
    opt.zero_grad(set_to_none=True);loss=objective(model(x),y,v);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
   rec=dict(epoch=epoch,loss=float(np.mean(losses)),validation_Dice=None,threshold=None)
   if epoch%2==0 or epoch==args.epochs:
    maps=val.column(infer,3,2);row,grid=p.calibrate(val,maps);rec.update(validation_Dice=row['Dice'],threshold=row['threshold'])
    if row['Dice']>best+1e-5:
     best=row['Dice'];torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=config),out/'best.pt');p.save_maps(out/'probabilities/val',maps);p.js(out/'selection.json',dict(epoch=epoch,**row));p.csvout(out/'thresholds.csv',grid)
   hist.append(rec);p.csvout(out/'history.csv',hist);print('V9_TRAIN',args.model,rec,flush=True)
  model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)['state_dict']);p.js(out/'training_run.json',dict(**config,epochs_completed=args.epochs,optimizer_updates=args.epochs*16,selected= json.loads((out/'selection.json').read_text()),selected_checkpoint_SHA256=p.sha(out/'best.pt'),elapsed_seconds=time.monotonic()-start))
 for col in [0,1,2,4]:
  p.save_maps(out/'probabilities'/('test' if col==4 else 'train')/('C'+str(col)),val.column(infer,col,2));print('V9_COLUMN',args.model,col,flush=True)
 p.save_maps(out/'probabilities/upper',full_maps(validated,infer,384 if args.model=='unet' else 256));print('V9_INFERENCE_DONE',args.model,flush=True)
if __name__=='__main__':main()
