#!/usr/bin/env python3
"""Root/background only. Checkpoint and threshold selection never use C4."""
import argparse,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import common as c
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_uploaded_upper as original
from train_segformer_progress import RootSegFormer

class RefinedSegFormer(nn.Module):
    def __init__(self):
        super().__init__();self.base=RootSegFormer()
        self.refine=nn.Sequential(nn.Conv2d(4,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1,bias=False),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.refine[-1].weight);nn.init.zeros_(self.refine[-1].bias)
    def forward(self,x):
        logit=self.base(x);return logit+self.refine(torch.cat([x,logit],1))

def create(name): return original.SmallUNet(12) if name=='unet' else RefinedSegFormer()
def erode(x): return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
def soft_skel(x,n=5):
    opened=F.max_pool2d(erode(x),3,1,1);s=F.relu(x-opened)
    for _ in range(n):
        x=erode(x);d=F.relu(x-F.max_pool2d(erode(x),3,1,1));s=s+F.relu(d-s*d)
    return s
def loss_fn(logit,y,v,variant='balanced'):
    b=F.binary_cross_entropy_with_logits(logit,y,reduction='none');pos=b[(y>.5)&(v>.5)];neg=b[(y<.5)&(v>.5)]
    # Hard negatives explicitly penalise bubbles/cracks that resemble fine roots.
    k=max(1,int(neg.numel()*.2)) if variant=='balanced' else min(neg.numel(),max(1,pos.numel()*4+256*len(y)))
    hard=neg.topk(k).values;negative=hard.mean()
    positive=pos.mean() if pos.numel() else logit.sum()*0
    p=logit.sigmoid()*v;g=y*v
    dice=1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1)
    sp=soft_skel(p);sg=soft_skel(g)
    precision=((sp*g).sum()+1)/(sp.sum()+1);recall=((sg*p).sum()+1)/(sg.sum()+1)
    cl=1-2*precision*recall/(precision+recall+1e-7)
    bce=.5*(positive+negative) if variant=='balanced' else (pos.sum()+hard.sum())/max(1,pos.numel()+hard.numel())
    return bce+dice+.15*cl

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',choices=['unet','segformer'],required=True)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--epochs',type=int,default=90);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--lr',type=float,default=.0003);p.add_argument('--evaluate',choices=['test','train','upper'])
    p.add_argument('--loss',choices=['balanced','pixel-hard'],default='balanced');p.add_argument('--output-name')
    a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(c.SEED);np.random.seed(c.SEED);random.seed(c.SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');data=c.Data(a.run);out=a.run/(a.output_name or a.model);out.mkdir(parents=True,exist_ok=True)
    model=create(a.model).to(device)
    def infer(ims):
        model.eval()
        with torch.inference_mode():
            x=torch.from_numpy(ims.transpose(0,3,1,2).copy()).to(device)
            return model(x).sigmoid()[:,0].cpu().numpy()
    if a.evaluate:
        assert (a.run/'selection.json').exists(),'Freeze validation-only model selection before evaluating C4'
        ck=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'])
        import json
        run=json.loads((out/'training_run.json').read_text());threshold=run['frozen_threshold']
        if a.evaluate=='upper':
            from PIL import Image
            folder=out/'predictions/upper';folder.mkdir(parents=True,exist_ok=True)
            for t,rgb in data.rgb.items():
                h,w=rgb.shape[:2];xs=list(range(0,w-255,192));ys=list(range(0,h-255,192))
                if xs[-1]!=w-256:xs.append(w-256)
                if ys[-1]!=h-256:ys.append(h-256)
                coords=[(x,y) for y in ys for x in xs];sm=np.zeros((h,w),np.float32);weights=np.zeros_like(sm)
                han=np.hanning(256).astype(np.float32);blend=np.maximum(han[:,None]*han[None,:],.02)
                for i in range(0,len(coords),a.batch):
                    xy=coords[i:i+a.batch];ims=np.stack([rgb[y:y+256,x:x+256] for x,y in xy]).astype(np.float32)/255
                    for (x,y),pr in zip(xy,infer(ims)):sm[y:y+256,x:x+256]+=pr*blend;weights[y:y+256,x:x+256]+=blend
                assert (weights>0).all();pr=sm/weights
                np.save(folder/(t+'_probability.npy'),pr)
                Image.fromarray((pr>=threshold).astype(np.uint8)*255).save(folder/(t+'_root_mask.png'))
            print('ROOT_UPPER_DONE',a.model,flush=True);return
        cols=[4] if a.evaluate=='test' else [0,1,2];reports=[]
        for col in cols:
            maps=data.column(infer,col,a.batch);c.save_maps(out/'predictions'/a.evaluate/('C'+str(col)),maps)
            reports.append(data.evaluate(maps,col,threshold))
        total=sum((np.array([r['root'][k] for k in ['TP','FP','FN','TN']]) for r in reports),np.zeros(4,np.int64))
        c.js(out/(a.evaluate+'_metrics.json'),dict(root=c.metrics(total),columns=reports,threshold=threshold))
        print('ROOT_EVALUATED',a.model,a.evaluate,c.metrics(total),flush=True);return
    old=torch.load(a.initial,map_location=device,weights_only=True)
    (model if a.model=='unet' else model.base).load_state_dict(old['state_dict'])
    config=dict(task='binary root/background ONLY',model=a.model,input_channels=3,output_channels=1,previous_period_input=False,
        temporal_loss=False,death_class=False,initial_checkpoint_sha256=c.sha(a.initial),
        initialization=('Second-stage binary-only refinement from four-period root-only checkpoint' if 'config' in old and old['config'].get('task')=='binary root/background ONLY' else 'Transfer from previously trained T1/T2 root-only model; same training columns, no C4 fitted labels'),
        initialization_path=str(a.initial),RGB_normalization='RGB/255',device=str(device),cuda_available=torch.cuda.is_available(),
        seed=c.SEED,epochs_max=a.epochs,batch_size=a.batch,samples_per_epoch=48,threads=a.threads,learning_rate=a.lr,
        loss=('0.5 positive BCE +0.5 hardest20% negative BCE +soft Dice +0.15 soft clDice' if a.loss=='balanced' else 'Equal-pixel BCE on positives and hardest(4*positive+256*batch) negatives +soft Dice +0.15 soft clDice'),loss_variant=a.loss,
        parameters=sum(x.numel() for x in model.parameters()),
        refiner='SegFormer only: native full-resolution RGB+root logit 4->16->16->1 residual; zero output initialization' if a.model=='segformer' else None,
        labels_modified=False,patch_size=256,manifest_sha256=c.sha(a.run/'dataset/manifest.csv'),
        provenance_sha256=c.sha(a.run/'dataset/provenance.json'),
        selection='Maximum validation strict root Dice at threshold0.5; evaluate every3 epochs, patience10 checks after minimum36 epochs',
        test_used_during_training_or_selection=False,torch_version=str(torch.__version__))
    c.js(out/'config.json',config);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=a.lr*.05)
    best=-1;bad=0;hist=[];start=time.monotonic()
    for epoch in range(1,a.epochs+1):
        tic=time.monotonic();model.train();losses=[]
        for i in range(0,48,a.batch):
            samples=[data.sample(j) for j in range(i,min(i+a.batch,48))]
            x=torch.from_numpy(np.stack([z[0] for z in samples]).transpose(0,3,1,2).copy()).to(device)
            y=torch.from_numpy(np.stack([z[1] for z in samples])[:,None]).to(device)
            v=torch.from_numpy(np.stack([z[2] for z in samples])[:,None]).to(device)
            opt.zero_grad(set_to_none=True);loss=loss_fn(model(x),y,v,a.loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step();losses.append(float(loss.detach()))
        rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_Dice=None,validation_Recall=None,validation_Precision=None,seconds=time.monotonic()-tic)
        if epoch%3==0 or epoch==a.epochs:
            maps=data.column(infer,3,a.batch);m=data.evaluate(maps,3,.5)['root']
            rec.update(validation_Dice=m['Dice'],validation_Recall=m['Recall'],validation_Precision=m['Precision'])
            if m['Dice']>best+1e-5:
                best=m['Dice'];bad=0
                torch.save(dict(state_dict=model.state_dict(),epoch=epoch,model=a.model,config=config,validation_Dice_at_05=best),out/'best.pt')
            else:bad+=1
            print('ROOT_EPOCH',a.model,epoch,'DICE',round(m['Dice'],4),'RECALL',round(m['Recall'],4),flush=True)
        hist.append(rec);c.csvout(out/'history.csv',hist);schedule.step()
        if epoch>=36 and bad>=10:break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'])
    maps=data.column(infer,3,a.batch);threshold,thresholds=c.calibrate(data,maps,3)
    c.csvout(out/'validation_thresholds.csv',thresholds);c.save_maps(out/'predictions/val/C3',maps)
    c.js(out/'validation_metrics.json',data.evaluate(maps,3,threshold))
    c.js(out/'training_run.json',dict(**config,epochs_completed=len(hist),best_epoch=ck['epoch'],
        frozen_threshold=threshold,checkpoint_sha256=c.sha(out/'best.pt'),elapsed_seconds=time.monotonic()-start))
    print('ROOT_TRAIN_DONE',a.model,data.evaluate(maps,3,threshold)['root'],flush=True)
if __name__=='__main__':main()
