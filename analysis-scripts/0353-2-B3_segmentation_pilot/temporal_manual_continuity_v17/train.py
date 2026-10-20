"""Fine-tune V14 starts on human temporal classes with U-Net width and routed continuity."""
import argparse,time
from pathlib import Path
import numpy as np
import torch
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--epochs',type=int,default=32);a=p.parse_args();torch.set_num_threads(3);torch.manual_seed(c.SEED);rng=np.random.default_rng(c.SEED);pick=np.random.default_rng(c.SEED);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported();out=a.run/a.model;out.mkdir(exist_ok=True);assert not(out/'training_run.json').exists(),'New run required'
start=time.monotonic();data=c.Data(a.run,a.roots,a.model);model=c.load_head(a.previous/a.model/'best_temporal_head.pt',device,warm=False);freq=data.counts();weights=np.clip(np.sqrt(freq.max()/np.maximum(freq,1)),1,8).astype('float32');tw=torch.tensor(weights,device=device);root=a.roots/a.model/('best_weights_fp16.npz' if a.model=='deeplab' else 'best.pt')
config=dict(version='temporal_manual_continuity_v17',model=a.model,root_architecture=c.MODEL_NAMES[a.model],root_stage_frozen=True,root_framework='TensorFlow1.12' if a.model=='deeplab' else 'PyTorch',temporal_framework='PyTorch',temporal_input_channels=31,root_SHA256=c.sha(root),warm_start_SHA256=c.sha(a.previous/a.model/'best_temporal_head.pt'),warm_start='V14 temporal checkpoint for this model; 25 input channels expanded to31 with zero new-channel weights',original_human_progress_SHA256=c.sha(a.run/'data/original_human_progress.json'),active_progress_SHA256=c.sha(a.run/'data/active_progress.json'),reference='Original manual temporal classes on U-Net morphology; added uncertain roots untyped',original_human_retained_separately=True,teacher_temporal_predictions_used=False,manual_typed_pixels_reclassified=0,junction_router='Training-only manual-semantic affinity; weak instance supervision',additional_loss='0.25 routed same-manual-class continuity across junctions',epochs_max=a.epochs,batch_size=4,samples_per_epoch=64,seed=c.SEED,split_seed=20261018,train_validation_test_pairs=[150,15,18],optimizer='AdamW lr1e-4 weight_decay1e-4; cosine eta_min1e-5',objective='Weighted CE +0.5classDice +0.2boundary BCE +0.05same-label local consistency +0.3same-class skeleton-run loss',class_counts=freq.tolist(),class_weights=weights.tolist(),axis_context='Full-frame RGB/width/tangent tracing and previous-root correspondence before cropping',checkpoint_selection='Maximum Manual-temporal/morphology-reference validation full-frame whole-axis macroDice, every4epochs; epoch0eligible; original audit not used for selection',decoding='Full-frame blended logits then whole-axis grammar, not independent tile labels',historical_same_field_exposure=True,independent_external_test_images=0,death_included=False,device=str(device),CUDA=torch.cuda.is_available(),arithmetic='BF16 autocast / FP32 objective' if amp else 'FP32')
c.js(out/'config.json',config)
def save(file,epoch):torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=config,class_multipliers=getattr(model,'class_multipliers',[1.,1.,1.]),class_adjustment=getattr(model,'class_adjustment',[1.,1.,1.])),file)

def route_patch(data,row,aug):
 x,y=int(row['x']),int(row['y']);arr=data.axes[row['current']]['axis_map'][y:y+256,x:x+256];arr=np.rot90(arr,aug['rotation_quarters'])
 if aug['flip_x']:arr=np.fliplr(arr)
 if aug['flip_y']:arr=np.flipud(arr)
 return arr.copy()
def route_loss(logits,gt,route):
 probability=logits.softmax(1);terms=[]
 for b in range(len(gt)):
  ids=route[b];valid=(gt[b]>0)&(gt[b]<255);unique=torch.unique(ids[valid]);used=0
  for axis_id in unique:
   if axis_id<=0:continue
   for k in [1,2,3]:
    pos=torch.nonzero((ids==axis_id)&(gt[b]==k),as_tuple=False)
    if len(pos)<8:continue
    step=max(1,len(pos)//64);pos=pos[::step][:64];q=probability[b,:,pos[:,0],pos[:,1]];mean=q.mean(1,keepdim=True)
    terms.append(-torch.log(mean[k-1,0].clamp_min(1e-6))+.5*(q-mean).square().sum(0).mean());used+=1
   if used>=32:break
 return torch.stack(terms).mean() if terms else logits.sum()*0
from structured import decode
strength=1.0
def full_validation():
 from predict import evaluate
 maps={}
 for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
  pr=c.predict_probability(model,data.features[t],device).astype('float16').astype('float32')
  maps[t],_=decode(pr,data.axes[prev],data.axes[t],data.link[t],strength)
 scores=evaluate(data,maps,'val')
 return {ref:scores[ref]['end_to_end'] for ref in scores}
baseline=full_validation();c.js(out/'baseline_validation_whole_axis.json',baseline);best=baseline['revised']['macro_Dice'];best_epoch=0;save(out/'best_temporal_head.pt',0);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-5);queue=[];history=[];draws=[]
for epoch in range(1,a.epochs+1):
 tic=time.monotonic();model.train();losses=[]
 for step in range(0,64,4):
  rr=[]
  for j in range(4):
   if not queue:queue=pick.permutation(len(data.train)).tolist()
   rr.append(data.train[queue.pop()])
  items=[data.patch(row,True,rng) for row in rr];xx=torch.from_numpy(np.stack([x[0] for x in items])).to(device).contiguous(memory_format=torch.channels_last);yy=torch.from_numpy(np.stack([x[1] for x in items])).to(device);opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):z,b=model.forward_with_boundary(xx)
  mul=np.asarray(getattr(model,'class_multipliers',[1.,1.,1.]))*np.asarray(getattr(model,'class_adjustment',[1.,1.,1.]));logits=z.float()+torch.tensor(np.log(mul),device=device,dtype=torch.float32)[None,:,None,None]
  routed=np.stack([route_patch(data,row,item[2]) for row,item in zip(rr,items)]);axis_target=torch.from_numpy(routed).to(device)
  value=c.loss(logits,yy,tw,b.float(),xx[:,17])+.25*route_loss(logits,yy,axis_target);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(value.detach()))
  for j,(row,item) in enumerate(zip(rr,items)):draws.append(dict(epoch=epoch,draw=step+j,pair_id=row['pair_id'],tile_id=row['tile_id'],previous=row['previous'],current=row['current'],x=row['x'],y=row['y'],**item[2]))
 schedule.step();record=dict(epoch=epoch,train_loss=float(np.mean(losses)),learning_rate=opt.param_groups[0]['lr'],validation_revised_macro_Dice=None,validation_original_macro_Dice=None,seconds=time.monotonic()-tic)
 if epoch%4==0:
  report=full_validation();score=report['revised']['macro_Dice'];record.update(validation_revised_macro_Dice=score,validation_original_macro_Dice=report['original']['macro_Dice'])
  if score>best+1e-6:best=score;best_epoch=epoch;save(out/'best_temporal_head.pt',epoch)
 history.append(record);c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',draws);print('EPOCH',a.model,epoch,round(record['train_loss'],4),record['validation_revised_macro_Dice'],flush=True)
save(out/'last_temporal_head.pt',a.epochs);assert c.sha(root)==config['root_SHA256'];c.js(out/'training_run.json',dict(**config,epochs_completed=a.epochs,optimizer_updates=len(draws)//4,unique_training_pairs=len({x['pair_id'] for x in draws}),selected_epoch=best_epoch,selected_validation_whole_axis_macro_Dice=best,head_SHA256=c.sha(out/'best_temporal_head.pt'),seconds=time.monotonic()-start));print('TRAINED',a.model,best_epoch,round(best,4),flush=True)
