#!/usr/bin/env python3
"""ImageNet ResNet18 U-Net, native384 context and full-resolution RGB skip; no time inputs."""
import argparse,random,time,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import protocol as p

class Conv(nn.Sequential):
 def __init__(self,a,b):super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.ReLU())
class RootUNet(nn.Module):
 def __init__(self,initial=None):
  super().__init__();self.encoder=resnet18(weights=None)
  if initial:self.encoder.load_state_dict(torch.load(initial,map_location='cpu',weights_only=True))
  self.encoder.fc=nn.Identity();self.d3=Conv(768,128);self.d2=Conv(256,64);self.d1=Conv(128,32);self.d0=Conv(96,32)
  self.native=Conv(3,16);self.finish=Conv(48,16);self.head=nn.Conv2d(16,1,1)
  nn.init.constant_(self.head.bias,-3.)
  self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1));self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
 def train(self,mode=True):
  super().train(mode)
  for m in self.encoder.modules():
   if isinstance(m,nn.BatchNorm2d):m.eval()
  return self
 def forward(self,x):
  e=self.encoder;z=(x-self.mean)/self.std;s0=e.relu(e.bn1(e.conv1(z)));s1=e.layer1(e.maxpool(s0));s2=e.layer2(s1);s3=e.layer3(s2);s4=e.layer4(s3)
  def up(a,b):return torch.cat([F.interpolate(a,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1)
  z=self.d3(up(s4,s3));z=self.d2(up(z,s2));z=self.d1(up(z,s1));z=self.d0(up(z,s0));z=up(z,self.native(x))
  return self.head(self.finish(z))
def erode(x):return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
def skel(x):
 s=F.relu(x-F.max_pool2d(erode(x),3,1,1))
 for _ in range(5):
  x=erode(x);d=F.relu(x-F.max_pool2d(erode(x),3,1,1));s=s+F.relu(d-s*d)
 return s
def objective(logit,y,v):
 b=F.binary_cross_entropy_with_logits(logit,y,reduction='none');positive=b[(y>.5)&(v>.5)];negative=b[(y<.5)&(v>.5)]
 k=min(negative.numel(),max(1,4*positive.numel()+256*len(y)));hard=negative.topk(k).values
 bce=(positive.sum()+hard.sum())/max(1,positive.numel()+hard.numel());prob=logit.sigmoid()*v;gt=y*v
 dice=1-(2*(prob*gt).sum()+1)/(prob.sum()+gt.sum()+1);sp=skel(prob);sg=skel(gt)
 pr=((sp*gt).sum()+1)/(sp.sum()+1);re=((sg*prob).sum()+1)/(sg.sum()+1)
 return bce+dice+.15*(1-2*pr*re/(pr+re+1e-7))
def inference(model,device,tta=False):
 def infer(ims):
  model.eval();x=torch.from_numpy(ims.transpose(0,3,1,2).copy()).to(device)
  with torch.inference_mode():
   probs=model(x).sigmoid()
   if tta:
    for axes in [(2,),(3,),(2,3)]:probs+=model(x.flip(axes)).sigmoid().flip(axes)
    probs/=4
  return probs[:,0].cpu().numpy()
 return infer
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--initial',type=Path,required=True)
 a.add_argument('--epochs',type=int,default=75);a.add_argument('--threads',type=int,default=3);a.add_argument('--benchmark',action='store_true');args=a.parse_args()
 args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads);torch.manual_seed(p.SEED);np.random.seed(p.SEED);random.seed(p.SEED)
 assert p.sha(args.initial)=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
 data=p.Data(args.data);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=RootUNet(args.initial).to(device)
 enc=list(model.encoder.parameters());encids={id(z) for z in enc};dec=[z for z in model.parameters() if id(z) not in encids]
 opt=torch.optim.AdamW([{'params':enc,'lr':2e-5},{'params':dec,'lr':3e-4}],weight_decay=1e-4)
 config=dict(task='RGB-only root/background',model='U-Net with ImageNet ResNet18 encoder and full-resolution RGB skip',
  input_channels=3,output_channels=1,patch_size=384,samples_per_epoch=48,batch_size=2,seed=p.SEED,epochs_max=args.epochs,
  device=str(device),cuda_available=torch.cuda.is_available(),parameters=sum(z.numel() for z in model.parameters()),
  encoder_lr=2e-5,decoder_lr=3e-4,encoder_freeze_epochs=5,encoder_BN_stats='Frozen ImageNet running statistics; affine trainable after warm-up',
  initial_sha256=p.sha(args.initial),dataset_provenance_sha256=p.sha(args.data/'dataset/provenance.json'),
  objective='Hard-negative pixel BCE +soft Dice +0.15 soft clDice; human masks unchanged',
  validation='C3 native pixels, overlap384 windows with Hann weighting; no RGB outside C3',
  checkpoint='Maximum validation strict Dice at0.5, every5 epochs; patience5 checks after minimum35epochs',
  old_C4_status='Previously inspected same-position benchmark, not a fresh independent test; excluded from fitting/selection',
  temporal_input=False,temporal_loss=False,death_class=False)
 p.js(args.output/'config.json',config);hist=[];best=-1;bad=0;start=time.monotonic();infer=inference(model,device)
 for epoch in range(1,args.epochs+1):
  model.train()
  for z in enc:z.requires_grad_(epoch>5)
  losses=[];tic=time.monotonic();factor=.05+.95*.5*(1+np.cos(np.pi*(epoch-1)/args.epochs))
  for group,lr in zip(opt.param_groups,[2e-5,3e-4]):group['lr']=lr*factor
  for i in range(0,6 if args.benchmark else 48,2):
   ss=[data.sample(i+j) for j in range(2)];x=torch.from_numpy(np.stack([s[0] for s in ss]).transpose(0,3,1,2).copy()).to(device)
   y=torch.from_numpy(np.stack([s[1] for s in ss])[:,None]).to(device);v=torch.from_numpy(np.stack([s[2] for s in ss])[:,None]).to(device)
   opt.zero_grad(set_to_none=True);loss=objective(model(x),y,v);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
  if args.benchmark:print('BENCHMARK_3_STEPS_SECONDS',time.monotonic()-tic,flush=True);return
  rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_Dice=None,validation_Precision=None,validation_Recall=None,seconds=time.monotonic()-tic)
  if epoch%5==0 or epoch==args.epochs:
   maps=data.column(infer,3);m=data.evaluate(maps,3,.5)['root'];rec.update(validation_Dice=m['Dice'],validation_Precision=m['Precision'],validation_Recall=m['Recall'])
   if m['Dice']>best+1e-5:
    best=m['Dice'];bad=0;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=config,validation_Dice=best),args.output/'best.pt')
   else:bad+=1
   print('REFINED_ROOT_EPOCH',epoch,'DICE',round(m['Dice'],4),'RECALL',round(m['Recall'],4),flush=True)
  hist.append(rec);p.csvout(args.output/'history.csv',hist)
  if epoch>=35 and bad>=5:break
 ck=torch.load(args.output/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'])
 choices=[]
 for tta in [False,True]:
  maps=data.column(inference(model,device,tta),3);chosen,rows=p.calibrate(data,maps)
  p.save_maps(args.output/('val_tta' if tta else 'val_identity'),maps);p.csvout(args.output/('thresholds_tta.csv' if tta else 'thresholds_identity.csv'),rows)
  choices.append(dict(tta=tta,threshold=chosen['threshold'],root={k:v for k,v in chosen.items() if k!='threshold'}))
 selected=max(choices,key=lambda r:r['root']['Dice']);p.js(args.output/'validation_inference_selection.json',dict(selected=selected,trials=choices,selection_uses_C4=False))
 p.js(args.output/'training_run.json',dict(**config,best_epoch=ck['epoch'],epochs_completed=len(hist),seconds=time.monotonic()-start,checkpoint_sha256=p.sha(args.output/'best.pt')))
 print('REFINED_ROOT_DONE',selected,flush=True)
if __name__=='__main__':main()
