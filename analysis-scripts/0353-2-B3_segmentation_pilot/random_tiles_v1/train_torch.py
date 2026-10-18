"""Fresh training, fixed random tile split, untouched test until selection is frozen."""
import argparse, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import common as c
from models import create, objective

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--model',choices=['unet','segformer'],required=True)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--threads',type=int,default=3); p.add_argument('--batch',type=int,default=2)
    p.add_argument('--benchmark',action='store_true'); p.add_argument('--predict-only',action='store_true'); a=p.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(c.SEED); np.random.seed(c.SEED); random.seed(c.SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); data=c.Data(a.run); out=a.run/a.model; out.mkdir(parents=True,exist_ok=True)
    model=create(a.model).to(device)
    def infer(ims):
        model.eval()
        with torch.inference_mode():
            x=torch.from_numpy(ims.transpose(0,3,1,2).copy()).to(device); p=model(x).sigmoid()
        return p[:,0].float().cpu().numpy()
    if a.predict_only:
        ck=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ck['state_dict'])
        c.full_predict(infer,a.run,out/'upper',c.read(out/'selection.json')['threshold'],a.batch); return
    config=dict(model=c.NAMES[a.model],task='root/background',seed=c.SEED,device=str(device),cuda_available=torch.cuda.is_available(),
        initialization='Entire network randomly initialized; no ImageNet or B3 checkpoints',pretrained=False,
        model_code_SHA256=c.sha(Path(__file__).with_name('models.py')),manifest_SHA256=c.sha(a.run/'data/manifest.csv'),
        parameters=sum(z.numel() for z in model.parameters()),epochs_max=a.epochs,samples_per_epoch=64,batch_size=a.batch,
        patch_size=256,optimizer='AdamW',learning_rate=.0003,weight_decay=.0001,
        loss='Hard-negative pixel BCE (4*positives+256*batch negatives) +soft Dice +0.15 soft clDice (5 iterations)',
        training_precision='CPU BF16 autocast, FP32 objective/weights' if device.type=='cpu' else 'FP32',evaluation_precision='FP32',
        augmentation='Rotations/flips and brightness/gamma in0.9..1.1, training tiles only; original saved RGB unchanged',
        selection='Maximum validation Dice at0.5, every2 epochs; patience12 checks after30 epochs; threshold on validation only',
        test_used_for_training_or_selection=False,labels_modified=False,RGB_resized=False,death_class=False,
        architecture='ResNet18 U-Net native RGB skip; batch norms trained from scratch' if a.model=='unet' else 'SegFormer-B0 native RGB+logit residual refinement',
        torch_version=str(torch.__version__),initial_random_state_sha256=None)
    # Hash the randomly initialized state without creating a second large checkpoint.
    import hashlib
    hh=hashlib.sha256()
    for name,z in model.state_dict().items(): hh.update(name.encode()); hh.update(z.cpu().numpy().tobytes())
    config['initial_random_state_sha256']=hh.hexdigest(); c.js(out/'config.json',config)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    tr=[r for r in data.rows if r['split']=='train']; queue=[]
    def next_rows():
        nonlocal queue
        rr=[]
        for _ in range(a.batch):
            if not queue: queue=tr[:]; random.shuffle(queue)
            rr.append(queue.pop())
        return rr
    best=-1; bad=0; hist=[]; draws=[]; start=time.monotonic()
    for epoch in range(1,a.epochs+1):
        tic=time.monotonic(); model.train(); losses=[]; factor=.1+.9*.5*(1+np.cos(np.pi*(epoch-1)/a.epochs))
        for g in opt.param_groups: g['lr']=.0003*factor
        for step in range(0,4 if a.benchmark else 64,a.batch):
            rr=next_rows(); xx,yy,vv=data.batch(rr,True)
            x=torch.from_numpy(xx.transpose(0,3,1,2).copy()).to(device); y=torch.from_numpy(yy[:,None]).to(device); v=torch.from_numpy(vv[:,None]).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cpu'): logit=model(x)
            loss=objective(logit,y,v); assert torch.isfinite(loss); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); losses.append(float(loss.detach()))
            draws.extend(dict(epoch=epoch,draw=step+j,image_id=r['image_id'],tile_id=r['tile_id']) for j,r in enumerate(rr))
        if a.benchmark: print('BENCHMARK',a.model,time.monotonic()-tic,flush=True); return
        rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_Dice=None,validation_Recall=None,validation_Precision=None,seconds=time.monotonic()-tic)
        if epoch%2==0 or epoch==a.epochs:
            maps=data.predict(infer,'val',a.batch); m,_=data.evaluate(maps,.5); rec.update(validation_Dice=m['Dice'],validation_Recall=m['Recall'],validation_Precision=m['Precision'])
            if m['Dice']>best+1e-5:
                best=m['Dice']; bad=0; torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=config,validation_Dice_at_05=best),out/'best.pt')
            else: bad+=1
        hist.append(rec); c.csvout(out/'history.csv',hist)
        print('RANDOM_TILE_EPOCH',a.model,epoch,'loss',round(rec['train_loss'],4),'val_Dice',rec['validation_Dice'],'sec',round(time.monotonic()-tic,1),flush=True)
        if epoch>=30 and bad>=12: break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ck['state_dict'])
    maps=data.predict(infer,'val',a.batch); selected,grid=c.calibrate(data,maps); c.csvout(out/'validation_thresholds.csv',grid)
    c.js(out/'selection.json',dict(selected,selected_epoch=ck['epoch'],checkpoint_SHA256=c.sha(out/'best.pt'),test_seen=False))
    c.save_probabilities(out/'probabilities/val',maps)
    test=data.predict(infer,'test',a.batch); c.save_probabilities(out/'probabilities/test',test)
    reports={}
    for split,pr in [('val',maps),('test',test)]:
        m,rr=data.evaluate(pr,selected['threshold']); reports[split]=m; c.csvout(out/(split+'_per_tile_metrics.csv'),rr)
    c.js(out/'metrics.json',reports); c.csvout(out/'training_draws.csv',draws)
    c.js(out/'training_run.json',dict(config,best_epoch=ck['epoch'],epochs_completed=len(hist),elapsed_seconds=time.monotonic()-start,
        checkpoint_SHA256=c.sha(out/'best.pt'),threshold=selected['threshold'],optimizer_updates=len(draws)//a.batch,
        unique_training_images=len({r['image_id'] for r in draws})))
    c.full_predict(infer,a.run,out/'upper',selected['threshold'],a.batch)
    print('MODEL_DONE',a.model,reports,flush=True)

if __name__=='__main__': main()
