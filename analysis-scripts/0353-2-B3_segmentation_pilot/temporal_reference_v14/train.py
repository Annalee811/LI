"""Continue three-model temporal heads on separate U-Net-derived references."""
import argparse,json,math,shutil,time
from pathlib import Path
import numpy as np
import torch
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);p.add_argument('--epochs',type=int,default=24);p.add_argument('--samples',type=int,default=64);p.add_argument('--threads',type=int,default=3);a=p.parse_args()
torch.set_num_threads(a.threads);torch.manual_seed(c.SEED);rng=np.random.default_rng(c.SEED);pick=np.random.default_rng(c.SEED);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cpu' and torch.cpu._is_avx512_bf16_supported();out=a.run/a.model;out.mkdir(exist_ok=True)
start=time.monotonic();data=c.Data(a.run,a.roots,a.model);model=c.load_head(a.previous/a.model/'evaluation_head.pt',device);freq=data.class_counts();weights=np.clip(np.sqrt(max(freq.max(),1)/np.maximum(freq,1)),1,6).astype('float32');tw=torch.tensor(weights,device=device)
root_weight=a.roots/a.model/('best_weights_fp16.npz' if a.model=='deeplab' else 'best.pt')
config=dict(version='temporal_reference_v14',model=a.model,root_architecture=c.MODEL_NAMES[a.model],task='Three temporal classes on frozen predicted root support; T1 deterministic baseline',classes=c.NAMES,background_removed_before_temporal_classification=True,root_stage='Frozen latest annotation_refinement_v1 root/background network; not temporally fine-tuned',root_checkpoint_SHA256=c.sha(root_weight),warm_start_head_SHA256=c.sha(a.previous/a.model/'evaluation_head.pt'),warm_start_split='Prior v13 evaluation head on different same-field C columns; historical pixel exposure possible',teacher_temporal_head='v13 U-Net final all-upper fitted head plus latest binary U-Net roots',teacher_reference='Model-derived revised reference, not independent manual truth',original_annotations_SHA256=c.sha(a.run/'data/original_human_progress.json'),active_progress_SHA256=c.sha(a.run/'data/active_progress.json'),input_features='25 full-field adjacent-date channels, then cropped; only model-predicted root RGB/probabilities/topology, never manual previous masks',feature_cache_dtype='float32',seed=c.SEED,split_seed=20261018,epochs_max=a.epochs,samples_per_epoch=a.samples,batch_size=4,optimizer='AdamW',learning_rate=.0001,weight_decay=.0001,scheduler='CosineAnnealingLR eta_min0.00001',loss='Weighted CE +0.5 classDice +0.2 auxiliary boundary BCE +0.05 within-class consistency +0.3 same-class skeleton-run loss',training_teacher_assigned_confidence_min=.5,training_only_on_student_detected_teacher_roots=True,class_counts_training=freq.tolist(),class_weights=weights.tolist(),chain_penalty=float(model.chain_penalty),chain_penalty_selected_this_round=False,checkpoint_selection='Maximum revised-validation end-to-end three-class macro Dice; original references audit only; test never selects',paired_train_val_test=[150,15,18],independent_external_test_images=0,heldout_is_not_new_blind_test=True,full_upper_refit_this_round=False,device=str(device),CUDA=torch.cuda.is_available(),arithmetic='CPU BF16 autocast with FP32 objective/weights' if amp else 'FP32',model_parameters=sum(z.numel() for z in model.parameters()))
c.js(out/'config.json',config)
def save(path,epoch):torch.save(dict(state_dict=model.state_dict(),epoch=epoch,config=config,class_multipliers=getattr(model,'class_multipliers',[1.,1.,1.]),class_adjustment=getattr(model,'class_adjustment',[1.,1.,1.]),chain_penalty=model.chain_penalty),path)
baseline={s:c.evaluate(model,data,s,device,amp=amp) for s in ['val','test']};c.js(out/'baseline_metrics.json',baseline);best=baseline['val']['revised']['end_to_end']['macro_Dice'];best_epoch=0;save(out/'best_temporal_head.pt',0);print('START',a.model,'teacher_validation_macro_Dice',best,'device',device,flush=True)
opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=.00001);history=[];draws=[];queue=[]
for epoch in range(1,a.epochs+1):
 tic=time.monotonic();model.train();losses=[]
 for step in range(0,a.samples,4):
  rr=[]
  for j in range(4):
   if not queue:queue=pick.permutation(len(data.train)).tolist()
   rr.append(data.train[queue.pop()])
  items=[data.patch(r,True,rng) for r in rr];xx=torch.from_numpy(np.stack([s[0] for s in items])).to(device).contiguous(memory_format=torch.channels_last);yy=torch.from_numpy(np.stack([s[1] for s in items])).to(device)
  opt.zero_grad(set_to_none=True)
  with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=amp):logits,boundary=model.forward_with_boundary(xx)
  mul=np.asarray(getattr(model,'class_multipliers',[1.,1.,1.]))*np.asarray(getattr(model,'class_adjustment',[1.,1.,1.]));calibrated=logits.float()+torch.tensor(np.log(mul),device=device,dtype=torch.float32)[None,:,None,None]
  value=c.loss(calibrated,yy,tw,boundary.float(),xx[:,17]);assert torch.isfinite(value);value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(value.detach()))
  for j,(r,it) in enumerate(zip(rr,items)):draws.append(dict(epoch=epoch,draw=step+j,pair_id=r['pair_id'],tile_id=r['tile_id'],previous=r['previous'],current=r['current'],x=r['x'],y=r['y'],**it[2]))
 schedule.step();rec=dict(epoch=epoch,train_loss=float(np.mean(losses)),learning_rate=opt.param_groups[0]['lr'],validation_revised_macro_Dice=None,validation_original_macro_Dice=None,seconds=time.monotonic()-tic)
 if epoch%4==0 or epoch==a.epochs:
  report=c.evaluate(model,data,'val',device,amp=amp);score=report['revised']['end_to_end']['macro_Dice'];rec.update(validation_revised_macro_Dice=score,validation_original_macro_Dice=report['original']['end_to_end']['macro_Dice']);save(out/f'candidate_epoch{epoch}.pt',epoch)
  if score>best+1e-6:best=score;best_epoch=epoch;save(out/'best_temporal_head.pt',epoch)
 history.append(rec);c.csvout(out/'history.csv',history);c.csvout(out/'training_draws.csv',draws);print('EPOCH',a.model,epoch,'loss',round(rec['train_loss'],4),'revised_val',rec['validation_revised_macro_Dice'],'seconds',round(rec['seconds'],1),flush=True)
save(out/'last_temporal_head.pt',a.epochs);model=c.load_head(out/'best_temporal_head.pt',device);reports={s:c.evaluate(model,data,s,device,out/'evaluation'/s,amp=amp) for s in ['val','test']};c.js(out/'metrics.json',reports)
c.js(out/'training_run.json',dict(**config,epochs_completed=a.epochs,optimizer_updates=len(draws)//4,unique_training_pairs=len({r['pair_id'] for r in draws}),selected_epoch=best_epoch,selected_revised_validation_macro_Dice=best,selected_head_SHA256=c.sha(out/'best_temporal_head.pt'),last_head_SHA256=c.sha(out/'last_temporal_head.pt'),seconds=time.monotonic()-start))
assert c.sha(root_weight)==config['root_checkpoint_SHA256'];print('MODEL_DONE',a.model,'selected_epoch',best_epoch,'revised_test_macro',reports['test']['revised']['end_to_end']['macro_Dice'],'original_test_macro',reports['test']['original']['end_to_end']['macro_Dice'],flush=True)
