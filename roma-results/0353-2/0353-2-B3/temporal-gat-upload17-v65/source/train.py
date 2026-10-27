"""Train graph refiners and same-input no-GAT controls, with resumable epochs."""
from pathlib import Path
import argparse,csv,json,os,time,hashlib,random,shutil
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.special import softmax
from metrics import binary_curve
from model import Model
from prepare import O,MODELS,SEED,js,csvout,sha

def atomic_json(path,d):
 p=Path(path);tmp=p.with_suffix('.json.tmp');js(tmp,d);os.replace(tmp,p)
def save(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pt.tmp');torch.save(obj,tmp);os.replace(tmp,p)
def measures(cm):
 a=[]
 for k in range(3):
  tp=cm[k,k];fn=cm[k].sum()-tp;fp=cm[:,k].sum()-tp;den=2*tp+fp+fn;a.append(dict(class_name=['persistent','newly_observed','tip_extension'][k],Precision=float(tp/max(tp+fp,1)),Recall=float(tp/max(tp+fn,1)),Dice=float(2*tp/max(den,1)),IoU=float(tp/max(tp+fp+fn,1)),support=int(tp+fn)))
 return dict(classes=a,macro_Dice=float(np.mean([r['Dice'] for r in a])),macro_IoU=float(np.mean([r['IoU'] for r in a])),confusion=cm.tolist())
def link_report(y,p,threshold=None):
 positive=y==1;score=p[:,1]
 if threshold is None:
  choices=np.linspace(.1,.9,33);ss=[]
  for t in choices:
   q=score>=t;tp=np.sum(q&positive);fp=np.sum(q&~positive);fn=np.sum(~q&positive);ss.append(2*tp/max(2*tp+fp+fn,1))
  threshold=float(choices[int(np.argmax(ss))])
 q=score>=threshold;tp=int(np.sum(q&positive));fp=int(np.sum(q&~positive));fn=int(np.sum(~q&positive));return dict(threshold=float(threshold),same_root_Precision=tp/max(tp+fp,1),same_root_Recall=tp/max(tp+fn,1),same_root_F1=2*tp/max(2*tp+fp+fn,1),same_root_AP=binary_curve(positive,score)['average_precision'],same_root_AUC=binary_curve(positive,score)['AUC'],n_pairs=len(y),positive_pairs=int(positive.sum()),relation_accuracy=float((p.argmax(1)==y).mean()))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--kind',choices=['gat','mlp'],required=True);a=parser.parse_args();out=O/(a.model+'_'+a.kind);out.mkdir(exist_ok=True)
 if (O/'STOPPED_BY_USER.json').exists():raise SystemExit('STOPPED_BY_USER marker present')
 if (out/'completion.json').exists():print('ALREADY_COMPLETE',a.model,a.kind,flush=True);return
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
 prep=json.loads((O/'data/preparation.json').read_text());geo=np.load(O/'data/geometry.npz');graph=np.load(O/'data/graph.npz');samples=np.load(O/'data/samples.npz');feature=np.load(O/f'data/{a.model}_features.npz');pairs=torch.from_numpy(graph['pairs']);attrs=torch.from_numpy(graph['edge_attr']);N=len(graph['node_split']);directions=np.r_[graph['pairs'],graph['pairs'][:,::-1],np.c_[np.arange(N),np.arange(N)]];direction_attrs=np.r_[graph['edge_attr'],graph['edge_attr']*np.array([1,1,1,-1,-1,1,1,1]),np.zeros((N,8))].astype(np.float32);edge_index=torch.from_numpy(directions.T.copy());edge_attr=torch.from_numpy(direction_attrs);x=torch.from_numpy(feature['x']);sample_nodes=torch.from_numpy(samples['node']);sample_y=torch.from_numpy(samples['target']);sample_log=torch.from_numpy(feature['sample_log']);node_split=graph['node_split'];s_split=node_split[samples['node']];pair_y=torch.from_numpy(graph['pair_target']);edge_split=graph['edge_split'];train_pairs=torch.from_numpy(np.flatnonzero(edge_split==0));train_samples=torch.from_numpy(np.flatnonzero(s_split==0))
 weights=np.array(prep['node_histogram_by_split'][0]);weights=np.clip(np.sqrt(weights.max()/np.maximum(weights,1)),1,5);weights=torch.from_numpy(weights.astype(np.float32));cc=np.bincount(graph['pair_target'][edge_split==0],minlength=3);relweights=torch.from_numpy(np.clip(np.sqrt(cc.max()/np.maximum(cc,1)),1,5).astype(np.float32))
 model=Model(x.shape[1],a.kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0002);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=.00002)
 config=dict(model=a.model,kind=a.kind,epochs=100,seed=SEED,source_SHA256=prep['source_SHA256'],input_features=prep['feature_names'],root_IDs_never_input=True,backbones_and_prior_head_frozen=True,initialization='Random new graph/refinement and association heads; zero temporal residual starts at archived head probabilities',training='Full transductive feature graph, losses only on training families; test labels never used for checkpoint or threshold',classifier_scope='Persistent / newly observed / tip extension; T1 remains baseline',objective='Weighted sampled-pixel CE + 0.12 soft Dice + 0.35 relation CE + 0.003 temporal-residual penalty',selection='0.75 validation class macro Dice + 0.25 validation same-root pair F1; epochs1..100 only',parameter_count=sum(p.numel() for p in model.parameters()),optimizer='AdamW .002 -> .00002 cosine',temporal_weights=weights.tolist(),relation_weights=relweights.tolist())
 atomic_json(out/'config.json',config);first=1;history=[];best=-1.;elapsed0=0.;tic=time.monotonic()
 if (out/'resume.pt').exists():
  ck=torch.load(out/'resume.pt',weights_only=False,map_location='cpu');assert ck['config']==config;model.load_state_dict(ck['state_dict']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);torch.set_rng_state(ck['torch_rng']);first=ck['epoch']+1;history=ck['history'];best=ck['best'];elapsed0=ck['elapsed_seconds'];print('RESUMED',a.model,a.kind,first,flush=True)
 else:save(out/'initial.pt',dict(state_dict=model.state_dict(),epoch=0,config=config))
 def forward():return model(x,edge_index,edge_attr,pairs,attrs)
 def validation(delta,relation):
  s=np.flatnonzero(s_split==1);p=(sample_log[s]+delta[sample_nodes[s]]).softmax(1).cpu().numpy();yy=samples['target'][s];cm=np.bincount(yy*3+p.argmax(1),minlength=9).reshape(3,3);m=measures(cm);e=np.flatnonzero(edge_split==1);link=link_report(graph['pair_target'][e],relation[e].softmax(1).cpu().numpy());return m,link
 for epoch in range(first,101):
  if (O/'STOPPED_BY_USER.json').exists():atomic_json(out/'status.json',dict(state='stopped',epoch_completed=epoch-1));return
  te=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);delta,relation,z=forward();pred=sample_log[train_samples]+delta[sample_nodes[train_samples]];loss_ce=F.cross_entropy(pred,sample_y[train_samples],weight=weights);prob=pred.softmax(1);gt=F.one_hot(sample_y[train_samples],3).float();dice=1-((2*(prob*gt).sum(0)+1)/(prob.sum(0)+gt.sum(0)+1)).mean();linkloss=F.cross_entropy(relation[train_pairs],pair_y[train_pairs],weight=relweights);penalty=delta.square().mean();loss=loss_ce+.12*dice+.35*linkloss+.003*penalty;assert torch.isfinite(loss);loss.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();lr=optimizer.param_groups[0]['lr'];scheduler.step();model.eval()
  with torch.inference_mode():vd,vr,_=forward();vm,vl=validation(vd,vr)
  score=.75*vm['macro_Dice']+.25*vl['same_root_F1'];rec=dict(epoch=epoch,training_loss=float(loss.detach()),training_temporal_CE=float(loss_ce.detach()),training_relation_CE=float(linkloss.detach()),validation_macro_Dice=vm['macro_Dice'],validation_same_root_F1=vl['same_root_F1'],validation_score=score,learning_rate=lr,seconds=time.monotonic()-te);history.append(rec)
  if score>best:
   best=score;save(out/'best.pt',dict(state_dict=model.state_dict(),epoch=epoch,config=config,validation_temporal=vm,validation_links=vl,validation_score=score))
  elapsed=elapsed0+time.monotonic()-tic;save(out/'resume.pt',dict(state_dict=model.state_dict(),epoch=epoch,config=config,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),torch_rng=torch.get_rng_state(),history=history,best=best,elapsed_seconds=elapsed));csvout(out/'history.csv',history);atomic_json(out/'status.json',dict(stage='training',epoch_completed=epoch,epochs_total=100,last_epoch_seconds=rec['seconds'],elapsed_seconds=elapsed,pid=os.getpid(),best_validation_score=best,updated_unix=time.time()))
  if epoch==1 or epoch%10==0:print('EPOCH',a.model,a.kind,epoch,'loss',round(rec['training_loss'],5),'valDice',round(vm['macro_Dice'],5),'valLinkF1',round(vl['same_root_F1'],5),'sec',round(rec['seconds'],2),flush=True)
 selected=torch.load(out/'best.pt',weights_only=False,map_location='cpu');model.load_state_dict(selected['state_dict']);model.eval()
 with torch.inference_mode():delta,relation,z=forward()
 np.savez_compressed(out/'inference.npz',temporal_delta=delta.numpy(),relation_prob=relation.softmax(1).numpy(),embedding=z.numpy());test_edges=np.flatnonzero(edge_split==2);metrics=link_report(graph['pair_target'][test_edges],relation[test_edges].softmax(1).numpy(),threshold=selected['validation_links']['threshold']);atomic_json(out/'test_identity_metrics.json',metrics)
 atomic_json(out/'completion.json',dict(state='complete',epochs_completed=100,selected_epoch=selected['epoch'],selected_validation_temporal=selected['validation_temporal'],selected_validation_links=selected['validation_links'],test_identity=metrics,seconds=elapsed0+time.monotonic()-tic,checkpoint_SHA256=sha(out/'best.pt'),config=config));atomic_json(out/'status.json',dict(stage='complete',epoch_completed=100,selected_epoch=selected['epoch'],updated_unix=time.time()));print('COMPLETE',a.model,a.kind,'epoch',selected['epoch'],'testLinkF1',metrics['same_root_F1'],flush=True)
if __name__=='__main__':main()
