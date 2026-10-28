"""100 actual local-only training epochs; no global U-Net parameter is in the optimizer."""
import fcntl,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import LocalCorrector
from warmstart import inherit_latest_module
from backbone import pool_groups
from data import Data
from prepare import OUT,SEED,js,sha,csvout
from metrics import binary_curve

def save(p,v):q=p.with_suffix('.tmp');torch.save(v,q);os.replace(q,p)
def metrics(y,p):
 cm=np.bincount(y*3+p,minlength=9).reshape(3,3);rows=[]
 for k,s in enumerate(['persistent','newly_observed','tip_extension']):
  tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp);rows.append(dict(name=s,Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),Dice=2*tp/max(2*tp+fp+fn,1),IoU=tp/max(tp+fp+fn,1),support=int(cm[k].sum())))
 return dict(classes=rows,confusion=cm.tolist(),macro_Dice=float(np.mean([r['Dice'] for r in rows])),accuracy=float(np.trace(cm)/max(cm.sum(),1)))
def edit_stats(d,p):
 edit=p!=d['anchor'];wrong=d['target']!=d['anchor'];fixed=int((edit&wrong&(p==d['target'])).sum());harmed=int((edit&~wrong).sum());stillwrong=int((edit&wrong&(p!=d['target'])).sum());return dict(changed=int(edit.sum()),fixed=fixed,harmed=harmed,changed_but_still_wrong=stillwrong,net_corrected=fixed-harmed,edit_precision=fixed/max(int(edit.sum()),1))
def proposed(model,h,d):
 logits,edit=model.local_logits(h,d['node'],d['pf']);p=logits.softmax(1);conf=p.max(1).values*edit.sigmoid();label=p.argmax(1)
 # At candidate junctions, a proposed extension simply keeps the original label;
 # the module is not authorized to create extension at connection points.
 connection=d['pf'][:,18]>.5;eligible=(~connection)|(label!=2)
 return label.cpu().numpy(),conf.cpu().numpy(),eligible.cpu().numpy(),logits,edit

def calibrated(model,data,h,threshold=None,split='val'):
 d=data.sets[split];q,conf,allow,_,_=proposed(model,h,data.tensor[split]);best=None
 for th in ([threshold] if threshold is not None else np.r_[np.arange(.1,.96,.05),1.01]):
  pp=d['anchor'].copy();take=(conf>=th)&allow;pp[take]=q[take];s=edit_stats(d,pp)
  # Validation only; prioritize avoiding damage to already-correct U-Net labels.
  score=s['fixed']-3*s['harmed']-.5*s['changed_but_still_wrong']
  if best is None or score>best[0] or (score==best[0] and th>best[1]):best=(score,float(th),pp,s)
 return best

def main():
 OUT.mkdir(exist_ok=True);lock=(OUT/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (OUT/'completion.json').exists():return
 if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker')
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True);data=Data();model=LocalCorrector();inherit_latest_module(model);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.0003);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,100,eta_min=1e-5);tr=data.tensor['train'];classcounts=np.bincount(data.sets['train']['target'],minlength=3);cw=torch.tensor(np.sqrt(classcounts.max()/classcounts.clip(1)).clip(1,4),dtype=torch.float32)
 conf=json.loads((OUT/'preparation.json').read_text());config=dict(epochs=100,seed=SEED,source_sha256=conf['source_sha256'],scope='Only local candidate corrections, same-root/attachment supervision and explicit history-supported persistence evidence; no global pixel loss; source24 unchanged',U_Net_parameters_in_optimizer=False,old_ConvLSTM_GAT_weights_loaded=True,learned_module_selected_epoch=92,learned_module_checkpoint_sha256='47175a3ac53b2341a2461fb92227bbcdbfe3c2ab371f911fa459c2569f3f3728',frozen_base_selected_epoch=45,frozen_base_checkpoint_sha256='c523b9649828e0e8f9d0187231edf262bf9a8ece62e205e8696ae94f8ff100b5',base_V75_after_frozen=True,local_reference='24',selection='Validation fixed errors -3 harmed correct pixels -.5 edits remaining wrong; reference-free candidate mask',class_weights=cw.tolist(),model_source_sha256=sha(Path(__file__).with_name('model.py')));js(OUT/'config.json',config)
 def ck(epoch):return dict(epoch=epoch,state_dict=model.state_dict(),config=config)
 first=1;history=[];bestscore=-float('inf');elapsed0=0.;tic=time.monotonic()
 if (OUT/'resume.pt').exists():
  c=torch.load(OUT/'resume.pt',weights_only=False,map_location='cpu');assert c['config']==config;model.load_state_dict(c['state_dict']);opt.load_state_dict(c['optimizer']);sched.load_state_dict(c['scheduler']);torch.set_rng_state(c['torch_rng']);history=c['history'];first=c['epoch']+1;bestscore=c['bestscore'];elapsed0=c['elapsed_seconds']
 else:
  save(OUT/'initial.pt',ck(0));model.eval()
  with torch.inference_mode():
   h,_,_=data.forward(model);log,e=model.local_logits(h,tr['node'],tr['pf']);vv=data.tensor['val'];vl,_=model.local_logits(h,vv['node'],vv['pf']);js(OUT/'epoch0_metrics.json',dict(epoch=0,training_CE=float(F.cross_entropy(log,tr['target'],weight=cw)) if len(tr['target']) else 0.,validation_CE=float(F.cross_entropy(vl,vv['target'],weight=cw)) if len(vv['target']) else 0.))
 linktarget=torch.tensor(data.g['target'],dtype=torch.float32);branchix=np.flatnonzero((data.g['node_split']==0)&(data.branch>=0));by=torch.tensor(data.branch[branchix],dtype=torch.float32);bp=torch.tensor(min(float((by==0).sum()/max((by==1).sum(),1)),15.));edit_y=((tr['target']!=tr['anchor'])|(tr['weak_persistent']&(tr['anchor']!=0))|(tr['weak_new_attachment']&(tr['anchor']!=1))).float();rs=data.rs
 for epoch in range(first,101):
  if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('Stopped; latest complete epoch saved')
  st=time.monotonic();model.train();opt.zero_grad(set_to_none=True);h,link,z=data.forward(model);log,e=model.local_logits(h,tr['node'],tr['pf']);weak=tr['weak_persistent'];ce=F.cross_entropy(log,tr['target'],weight=cw,reduction='none') if len(tr['target']) else log.sum(1);newweak=tr['weak_new_attachment'];factor=torch.where((weak&(tr['target']!=0))|(newweak&(tr['target']!=1)),.25,1.);ce=(ce*factor).sum()/factor.sum().clamp_min(1);persistence=F.cross_entropy(log[weak],torch.zeros(int(weak.sum()),dtype=torch.int64)) if weak.any() else h.sum()*0;newattachment=F.cross_entropy(log[newweak],torch.ones(int(newweak.sum()),dtype=torch.int64)) if newweak.any() else h.sum()*0;editloss=F.binary_cross_entropy_with_logits(e,edit_y) if len(edit_y) else h.sum()*0
  links=[]
  for same in [True,False]:
   ix=np.flatnonzero((data.g['split']==0)&(data.same==same));ys=linktarget[ix];pw=torch.tensor(float((ys==0).sum()/max((ys==1).sum(),1)));v=F.binary_cross_entropy_with_logits(link[ix],ys,pos_weight=pw,reduction='none');hard=v[ys==0];links.append(v.mean()+.25*hard.topk(max(1,len(hard)//4)).values.mean())
  linkloss=torch.stack(links).mean();branchloss=F.binary_cross_entropy_with_logits(model.branch_logits(h,data.extra)[branchix],by,pos_weight=bp);state=model.root_state(pool_groups(h,rs['nodes'],rs['ptr']))[:,0];stateloss=F.binary_cross_entropy_with_logits(state,rs['target'].float())
  loss=ce+.5*persistence+.5*newattachment+.3*editloss+.35*linkloss+.15*branchloss+.1*stateloss;assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
  if epoch==1:
   gradients=dict(ConvLSTM=float(model.memory.gates.weight.grad.abs().sum()),GAT=[float(q.project.weight.grad.abs().sum()) for q in model.layers]);assert gradients['ConvLSTM']>0 and all(v>0 for v in gradients['GAT']);js(OUT/'gradient_verification.json',dict(passed=True,**gradients))
  opt.step();lr=opt.param_groups[0]['lr'];sched.step();model.eval()
  with torch.inference_mode():
   hh,ll,_=data.forward(model);vv=data.tensor['val'];vl,_=model.local_logits(hh,vv['node'],vv['pf']);score,th,pp,ed=calibrated(model,data,hh);met=metrics(data.sets['val']['target'],pp);vi=np.flatnonzero((data.g['split']==1)&data.same);ap=binary_curve(data.g['target'][vi]==1,ll[vi].sigmoid().numpy())['average_precision'];valce=float(F.cross_entropy(vl,vv['target'],weight=cw)) if len(vv['target']) else 0.
  rec=dict(epoch=epoch,training_total=float(loss.detach()),training_local_CE=float(ce.detach()),validation_local_CE=valce,training_connection_BCE=float(linkloss.detach()),training_branch_BCE=float(branchloss.detach()),training_persistence_evidence_CE=float(persistence.detach()),training_new_attachment_evidence_CE=float(newattachment.detach()),validation_score=score,validation_edit_threshold=th,validation_changed=ed['changed'],validation_fixed=ed['fixed'],validation_harmed=ed['harmed'],validation_local_macro_Dice=met['macro_Dice'],validation_within_connection_AP=ap,learning_rate=lr,seconds=time.monotonic()-st);history.append(rec)
  selection=score+.001*float(ap)-.00001*valce
  if selection>bestscore:bestscore=selection;save(OUT/'best.pt',dict(**ck(epoch),validation_threshold=th,validation_edits=ed,validation=met))
  elapsed=elapsed0+time.monotonic()-tic;save(OUT/'resume.pt',dict(**ck(epoch),optimizer=opt.state_dict(),scheduler=sched.state_dict(),torch_rng=torch.get_rng_state(),history=history,bestscore=bestscore,elapsed_seconds=elapsed));csvout(OUT/'history.csv',history);js(OUT/'status.json',dict(stage='training_local_only',epoch_completed=epoch,epochs_total=100,pid=os.getpid(),elapsed_seconds=elapsed,last_epoch_seconds=rec['seconds']))
  if epoch==1 or epoch%10==0:print('LOCAL EPOCH',epoch,'validation edits',ed,'seconds',rec['seconds'],flush=True)
 selected=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['state_dict']);model.eval()
 with torch.inference_mode():
  h,link,z=data.forward(model);score,th,p,ed=calibrated(model,data,h,selected['validation_threshold'],'test');met=metrics(data.sets['test']['target'],p)
 np.savez_compressed(OUT/'inference.npz',h=h.numpy(),connection=link.sigmoid().numpy(),embedding=z.numpy());js(OUT/'completion.json',dict(state='complete',epochs_completed=100,selected_epoch=selected['epoch'],edit_threshold=th,test_local_edits=ed,test_local_metrics=met,checkpoint_sha256=sha(OUT/'best.pt'),elapsed_seconds=elapsed0+time.monotonic()-tic));js(OUT/'status.json',dict(stage='complete',epoch_completed=100));print('LOCAL COMPLETE',ed,flush=True)
if __name__=='__main__':main()
