"""Continue a temporal head against a teacher proposal; original audit is separate."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
import common as c

def atomic_save(obj,path):
    temp=path.with_name(path.name+'.tmp');torch.save(obj,temp);os.replace(temp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--epochs',type=int,default=16);a=p.parse_args()
    out=a.run/a.model;out.mkdir(exist_ok=True)
    if (out/'training_run.json').exists():print('REUSE_COMPLETED_TRAINING',a.model,flush=True);return
    torch.set_num_threads(3);torch.manual_seed(c.SEED);rng=np.random.default_rng(c.SEED);pick=np.random.default_rng(c.SEED);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported();data=c.Data(a.run,a.roots,a.model)
    start=time.monotonic();warm=a.previous/a.model/'best_temporal_head.pt';model=c.load_head(warm,device,warm=True);model.class_multipliers=[1.,1.,1.];model.class_adjustment=[1.,1.,1.]
    freq=data.counts();weights=np.clip(np.sqrt(freq.max()/np.maximum(freq,1)),1,12).astype('float32');tw=torch.tensor(weights,device=device);root=a.roots/a.model/('best_weights_fp16.npz' if a.model=='deeplab' else 'best.pt')
    config=dict(version='unet_aligned_v21',model=a.model,reference='Full U-Net-derived proposal; validation selects teacher consistency, not independent accuracy',root_stage_frozen=True,root_framework='TensorFlow1.12' if a.model=='deeplab' else 'PyTorch',temporal_framework='PyTorch',warm_start_SHA256=c.sha(warm),root_SHA256=c.sha(root),progress_SHA256=c.sha(a.run/'data/selected_progress.json'),training_reference_SHA256=c.sha(a.run/'data/active_progress.json'),temporal_input_channels=31,epochs_max=a.epochs,seed=c.SEED,batch_size=4,samples_per_epoch=64,optimizer='AdamW lr=2e-5 weight_decay=1e-4 cosine eta_min=2e-6',objective='weighted CE +0.5 class Dice +0.2 boundary BCE +0.05 same-label local consistency +0.3 same-class skeleton-path loss',sampling='25% uniform,12.5% edited-window,12.5% persistent,25% new,25% extension; identical sampling/augmentation seeds across models',class_counts=freq.tolist(),class_weights=weights.tolist(),selection='Teacher-derived validation tile macro Dice consistency; every 2 epochs; starting checkpoint eligible; original annotation audited separately',historical_same_field_exposure=True,independent_external_test_images=0,death_included=False,device=str(device),arithmetic='CPU BF16 with FP32 objective' if amp else 'FP32',resume='Last epoch includes optimizer, scheduler, sampler and torch RNG states')
    c.js(out/'config.json',config);opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=2e-6)
    pools={k:[] for k in [1,2,3]};edited_pool=[]
    from PIL import Image
    edit_masks={t:np.array(Image.open(a.run/'data/revision'/(t+'_changed_pixels.png')))>0 for t in c.PERIODS[1:]}
    for i,r in enumerate(data.train):
        x,y=int(r['x']),int(r['y']);g=data.targets['current'][r['current']][y:y+256,x:x+256]
        if edit_masks[r['current']][y:y+256,x:x+256].any():edited_pool.append(i)
        for k in pools:
            if (g==k).any():pools[k].append(i)
    def weights_checkpoint(epoch):return dict(state_dict=model.state_dict(),epoch=epoch,config=config,class_multipliers=[1.,1.,1.],class_adjustment=[1.,1.,1.])
    last=out/'resume.pt';history=[];draws=[];first=1;queue=[];best_epoch=0;elapsed_before=0.
    if last.exists():
        ck=torch.load(last,map_location=device,weights_only=True);assert ck['config']==config,'Resume configuration changed';model.load_state_dict(ck['state_dict']);opt.load_state_dict(ck['optimizer']);schedule.load_state_dict(ck['scheduler']);rng.bit_generator.state=ck['augmentation_rng'];pick.bit_generator.state=ck['sampler_rng'];torch.set_rng_state(ck['torch_rng'].cpu());queue=ck['queue'];history=ck['history'];draws=ck['draws'];best=ck['best'];best_epoch=ck['best_epoch'];first=ck['epoch']+1;elapsed_before=ck['seconds'];print('RESUMED',a.model,ck['epoch'],flush=True)
    else:
        baseline=c.evaluate_tiles(model,data,'val',device,amp)['current'];best=baseline['macro_Dice'];c.js(out/'baseline_validation.json',baseline);atomic_save(weights_checkpoint(0),out/'best_temporal_head.pt')
    for epoch in range(first,a.epochs+1):
        tic=time.monotonic();model.train();losses=[]
        for step in range(0,64,4):
            rr=[]
            for j in range(4):
                if j==0:
                    if not queue:queue=[int(z) for z in pick.permutation(len(data.train))]
                    idx=queue.pop()
                elif j==1 and (step//4)%2==0 and edited_pool:
                    idx=int(pick.choice(edited_pool))
                else:
                    pool=pools[j];idx=int(pick.choice(pool)) if pool else int(pick.integers(len(data.train)))
                rr.append(data.train[idx])
            items=[data.patch(row,True,rng) for row in rr];xx=torch.from_numpy(np.stack([i[0] for i in items])).to(device).contiguous(memory_format=torch.channels_last);yy=torch.from_numpy(np.stack([i[1] for i in items])).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):z,b=model.forward_with_boundary(xx)
            value=c.loss(z.float(),yy,tw,b.float(),xx[:,17]);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(value.detach()))
            for j,(row,item) in enumerate(zip(rr,items)):draws.append(dict(epoch=epoch,draw=step+j,pair_id=row['pair_id'],tile_id=row['tile_id'],previous=row['previous'],current=row['current'],x=row['x'],y=row['y'],**item[2]))
        schedule.step();rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),learning_rate=opt.param_groups[0]['lr'],validation_macro_Dice=None,seconds=time.monotonic()-tic)
        if epoch%2==0 or epoch==a.epochs:
            val=c.evaluate_tiles(model,data,'val',device,amp)['current'];rec['validation_macro_Dice']=val['macro_Dice']
            if val['macro_Dice']>best+1e-6:best=val['macro_Dice'];best_epoch=epoch;atomic_save(weights_checkpoint(epoch),out/'best_temporal_head.pt')
        history.append(rec);c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',draws)
        seconds=elapsed_before+time.monotonic()-start;resume=dict(**weights_checkpoint(epoch),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),augmentation_rng=rng.bit_generator.state,sampler_rng=pick.bit_generator.state,torch_rng=torch.get_rng_state(),queue=queue,history=history,draws=draws,best=best,best_epoch=best_epoch,seconds=seconds);atomic_save(resume,last)
        c.js(out/'status.json',dict(stage='training',epoch=epoch,epochs_max=a.epochs,selected_epoch=best_epoch,best_validation_macro_Dice=best));print('EPOCH',a.model,epoch,'loss',round(rec['train_loss'],5),'val',rec['validation_macro_Dice'],'seconds',round(rec['seconds'],2),flush=True)
    assert c.sha(root)==config['root_SHA256'];c.js(out/'training_run.json',dict(**config,epochs_completed=a.epochs,optimizer_updates=len(draws)//4,unique_training_pairs=len({r['pair_id'] for r in draws}),selected_epoch=best_epoch,selected_validation_macro_Dice=best,head_SHA256=c.sha(out/'best_temporal_head.pt'),seconds=elapsed_before+time.monotonic()-start));print('TRAINED',a.model,best_epoch,best,flush=True)

if __name__=='__main__':main()
