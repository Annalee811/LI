import argparse,csv,hashlib,json,os,random,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from study_model import initialize,pool
W=Path(__file__).resolve().parents[1]
def js(p,d):q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
def save(p,d):q=p.with_suffix('.tmp');torch.save(d,q);q.replace(p)
def csvout(p,rows):
 with p.open('w',newline='',encoding='utf-8-sig')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(cm):
 rows=[]
 for k,name in enumerate(['persistent','new','extension']):
  tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp);den=2*tp+fp+fn;rows.append(dict(class_id=k+1,name=name,TP=tp,FP=fp,FN=fn,support=tp+fn,Precision=tp/(tp+fp)if tp+fp else None,Recall=tp/(tp+fn)if tp+fn else None,Dice=2*tp/den if den else None,IoU=tp/(tp+fp+fn)if tp+fp+fn else None))
 return dict(classes=rows,macro_Dice=float(np.mean([r['Dice']for r in rows if r['Dice']is not None])),macro_IoU=float(np.mean([r['IoU']for r in rows if r['IoU']is not None])),confusion=cm.tolist())
def dataset():
 g=dict(np.load(W/'data/graph.npz'));d={k:torch.from_numpy(v)for k,v in g.items()if k in ['x','extra','pairs','attrs','phase','centers','part','frames','partition_frame_masks']};p={t:dict(np.load(W/f'data/{t}_pixels.npz'))for t in ['T2','T3','T4']};allp={k:np.concatenate([p[t][k]for t in p])for k in ['node','pf','target','split']};allp={k:torch.from_numpy(v)for k,v in allp.items()};j={k:torch.from_numpy(v)for k,v in dict(np.load(W/'data/junctions.npz')).items()};rs={k:torch.from_numpy(v)for k,v in dict(np.load(W/'data/root_state.npz')).items()};groups=json.loads((W/'data/junction_groups.json').read_text());return g,d,p,allp,j,rs,groups
def jfeatures(z,j):
 e=F.normalize(pool(z,j['nodes'],j['ptr']),dim=1);return torch.cat([(e[j['a']]-e[j['b']]).abs(),e[j['a']]*e[j['b']],j['fixed']],1)
def assignment(groups,score,split):
 out=[]
 for g in groups:
  if g['split']!=split:continue
  v=[sum(float(score[g['pair_indices'][str(a)+','+str(b)]])for a,b in opt)for opt in g['options']];out.append(int(np.argmax(v))==g['target_option'])
 return dict(correct=int(sum(out)),total=len(out),accuracy=float(np.mean(out))if out else None)
def evaluate(net,h,p,split):
 cm=np.zeros((3,3),np.int64)
 for t,d in p.items():
  ids=np.flatnonzero(d['split']==split)
  for start in range(0,len(ids),32768):
   i=ids[start:start+32768];pred=net.pixel_logits(h,torch.from_numpy(d['node'][i]),torch.from_numpy(d['pf'][i])).argmax(1).numpy();cm+=np.bincount(d['target'][i]*3+pred,minlength=9).reshape(3,3)
 return metrics(cm)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['full','no_lstm','no_gat','no_gates','current_only'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--epochs',type=int,default=100);ap.add_argument('--smoke',action='store_true');ap.add_argument('--threads',type=int,default=1);a=ap.parse_args();torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.use_deterministic_algorithms(True);g,d,p,pix,j,rs,groups=dataset();net,pair=initialize(a.variant,a.seed);folder=W/('smoke'if a.smoke else'experiments')/a.variant/str(a.seed);folder.mkdir(parents=True,exist_ok=True);tr=torch.nonzero(pix['split']==0).flatten();freq=torch.bincount(pix['target'][tr],minlength=3).float();weights=(freq.max()/freq.clamp_min(1)).sqrt().clamp(1,8);ei=torch.from_numpy(np.flatnonzero((g['edge_split']==0)&(g['edge_target']>=0)));ey=torch.from_numpy(g['edge_target'][ei.numpy()]).float();epos=(ey==0).sum()/(ey==1).sum();ji=torch.nonzero(j['split']==0).flatten();jy=j['target'][ji];jpos=(jy==0).sum()/(jy==1).sum();ranking=[q for q in groups if q['split']==0];idx=np.zeros((len(ranking),3,2),np.int64);mask=np.zeros(idx.shape,np.float32)
 for k,q in enumerate(ranking):
  for m,opt in enumerate(q['options']):
   for n,(u,v)in enumerate(opt):idx[k,m,n]=q['pair_indices'][str(u)+','+str(v)];mask[k,m,n]=1
 idx=torch.from_numpy(idx);mask=torch.from_numpy(mask);labels=torch.tensor([q['target_option']for q in ranking]);opt=torch.optim.AdamW(list(net.parameters())+list(pair.parameters()),lr=3e-4,weight_decay=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-6);hist=[];best=-1.;first=1;config=dict(variant=a.variant,seed=a.seed,epochs=a.epochs,optimizer='AdamW',learning_rate=3e-4,weight_decay=1e-4,gradient_norm=5,pixel_samples_per_epoch=8192,selection='Maximum validation macro Dice; test evaluated once after checkpoint selection',initialization='Fresh shared random parameters; no V90 checkpoint warm start',frozen_upstream_prior='Released V66 probabilities; historical within-B3 exposure',training_domain='Partition-pure B3 nodes; no cross-partition graph edges; separate masked frame contexts',scope='Internal controlled module study, NOT independent generalization',active_parameters=sum(v.numel()for v in net.parameters())+sum(v.numel()for v in pair.parameters()),data_audit_SHA256=hashlib.sha256((W/'reports/data_audit.json').read_bytes()).hexdigest(),code_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_code_SHA256=hashlib.sha256((W/'code/study_model.py').read_bytes()).hexdigest());js(folder/'config.json',config)
 if(folder/'resume.pt').exists():
  ck=torch.load(folder/'resume.pt',weights_only=False);assert ck['config']==config;net.load_state_dict(ck['backbone']);pair.load_state_dict(ck['pair']);opt.load_state_dict(ck['optimizer']);schedule.load_state_dict(ck['scheduler']);torch.set_rng_state(ck['rng']);hist=ck['history'];best=ck['best'];first=ck['epoch']+1
 starttime=time.monotonic()
 for epoch in range(first,a.epochs+1):
  start=time.monotonic();net.train();pair.train();opt.zero_grad(set_to_none=True);h,links,z=net(d);rng=np.random.default_rng(a.seed+epoch*1009);ii=tr[torch.from_numpy(rng.choice(len(tr),min(8192,len(tr)),replace=False))];logits=net.pixel_logits(h,pix['node'][ii],pix['pf'][ii]);y=pix['target'][ii];ce=F.cross_entropy(logits,y,weight=weights);prob=logits.softmax(1);truth=F.one_hot(y,3).float();dice=1-((2*(prob*truth).sum(0)+1)/(prob.sum(0)+truth.sum(0)+1)).mean();identity=F.binary_cross_entropy_with_logits(links[ei],ey,pos_weight=epos);scores=pair(jfeatures(z,j));jb=F.binary_cross_entropy_with_logits(scores[ji],jy,pos_weight=jpos);rank=F.cross_entropy((scores[idx]*mask).sum(-1),labels);state=net.root_state(pool(h,rs['nodes'],rs['ptr']))[:,0];state_loss=F.binary_cross_entropy_with_logits(state,rs['target']);pr=state.sigmoid();con=F.relu(pr[rs['consistency'][:,1]]-pr[rs['consistency'][:,0]]).square().mean();loss=ce+.5*dice+.1*identity+.2*jb+.1*rank+.1*state_loss+.05*con;assert torch.isfinite(loss);loss.backward()
  if epoch==1:
   grad={name:float(sum(p.grad.abs().sum()for p in module.parameters()if p.grad is not None))for name,module in [('encoder',net.encoder),('memory',net.memory),('gat',net.layers)]if module is not None};js(folder/'gradient_check.json',grad)
  torch.nn.utils.clip_grad_norm_(list(net.parameters())+list(pair.parameters()),5,error_if_nonfinite=True);opt.step();schedule.step();net.eval();pair.eval()
  with torch.inference_mode():hv,lv,zv=net(d);val=evaluate(net,hv,p,1)
  record=dict(epoch=epoch,training_total=float(loss.detach()),training_class_CE=float(ce.detach()),training_class_Dice_loss=float(dice.detach()),training_identity_BCE=float(identity.detach()),training_junction_BCE=float(jb.detach()),validation_macro_Dice=val['macro_Dice'],validation_macro_IoU=val['macro_IoU'],learning_rate=opt.param_groups[0]['lr'],seconds=time.monotonic()-start)
  if val['macro_Dice']>best:
   best=val['macro_Dice'];save(folder/'best.pt',dict(epoch=epoch,backbone=net.state_dict(),pair=pair.state_dict(),config=config,validation=val))
  hist.append(record);csvout(folder/'training_history.csv',hist);save(folder/'resume.pt',dict(epoch=epoch,backbone=net.state_dict(),pair=pair.state_dict(),config=config,optimizer=opt.state_dict(),scheduler=schedule.state_dict(),rng=torch.get_rng_state(),history=hist,best=best));js(folder/'status.json',dict(stage='training',epochs_completed=epoch,epochs_total=a.epochs,seconds_last_epoch=record['seconds']))
  if epoch==1 or epoch%10==0:print(a.variant,a.seed,'EPOCH',epoch,'VAL_DICE',round(best,6),'SECONDS',round(record['seconds'],2),flush=True)
 ck=torch.load(folder/'best.pt',weights_only=False);net.load_state_dict(ck['backbone']);pair.load_state_dict(ck['pair']);net.eval();pair.eval()
 with torch.inference_mode():h,links,z=net(d);scores=pair(jfeatures(z,j));val=evaluate(net,h,p,1);test=evaluate(net,h,p,2) if not a.smoke else dict(macro_Dice=None,macro_IoU=None,classes=[],smoke_only=True)
 linkmask=(g['edge_split']==2)&(g['edge_target']>=0);lp=links[torch.from_numpy(linkmask)].sigmoid().numpy();ly=g['edge_target'][linkmask];tp=int(((lp>=.5)&(ly==1)).sum());fp=int(((lp>=.5)&(ly==0)).sum());fn=int(((lp<.5)&(ly==1)).sum());test['identity_link_F1']=2*tp/max(2*tp+fp+fn,1);test['identity_links']=int(linkmask.sum());test['junction_assignment']=assignment(groups,scores.numpy(),2);test['selection_used_test']=False
 result=dict(variant=a.variant,seed=a.seed,epochs_completed=a.epochs,selected_epoch=ck['epoch'],validation=val,test=test,parameters=config['active_parameters'],elapsed_seconds=time.monotonic()-starttime,scope=config['scope']);js(folder/'result.json',result);np.savez_compressed(folder/'inference.npz',h=h.numpy(),embedding=z.numpy(),connection=links.sigmoid().numpy(),junction_logits=scores.numpy());predfolder=folder/'predictions';predfolder.mkdir(exist_ok=True)
 from PIL import Image
 source=Path('/workspace/work/V90_all_microtubes_20261118/model_source/assets/base/unet/full')
 for t in ['T1','T2','T3','T4']:
  image=np.array(Image.open(source/(t+'_classes.png')))
  if t in p:
   q=p[t]
   with torch.inference_mode():
    for i in range(0,len(q['node']),32768):
     sl=slice(i,i+32768);v=net.pixel_logits(h,torch.from_numpy(q['node'][sl]),torch.from_numpy(q['pf'][sl])).argmax(1).numpy()+1;image.ravel()[q['pixel'][sl]]=v.astype(np.uint8)
  Image.fromarray(image).save(predfolder/(t+'_classes.png'))
 js(folder/'status.json',dict(stage='complete',epochs_completed=a.epochs,test_evaluated_after_selection=True));print('RUN_COMPLETE',a.variant,a.seed,'test',test['macro_Dice'],flush=True)
if __name__=='__main__':main()
