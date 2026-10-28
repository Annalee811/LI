"""Minority-focused training, true-ID edge supervision, atomic100-epoch recovery."""
import argparse,json,os,time,fcntl
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from prepare import OUT,BASE,MODELS,SEED,js,csvout,sha
from model import Model

def save(p,v):
    tmp=p.with_suffix('.tmp');torch.save(v,tmp);os.replace(tmp,p)
def measures(y,p):
    cm=np.bincount(y*3+p,minlength=9).reshape(3,3);rows=[]
    for k in range(3):
        tp=int(cm[k,k]);fn=int(cm[k].sum()-tp);fp=int(cm[:,k].sum()-tp)
        rows.append(dict(name=['persistent','newly_observed','tip_extension'][k],Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),Dice=2*tp/max(2*tp+fp+fn,1),IoU=tp/max(tp+fp+fn,1),support=int(cm[k].sum())))
    return dict(classes=rows,macro_Dice=float(np.mean([r['Dice'] for r in rows])),confusion=cm.tolist())
def links(y,p,threshold=.5):
    q=p>=threshold;tp=int((q&(y==1)).sum());fp=int((q&(y==0)).sum());fn=int((~q&(y==1)).sum())
    return dict(F1=2*tp/max(2*tp+fp+fn,1),Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),threshold=float(threshold),pairs=len(y),positive_pairs=int((y==1).sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--kind',choices=['gat','mlp'],required=True);a=ap.parse_args()
    folder=OUT/(a.model+'_'+a.kind);folder.mkdir(exist_ok=True);data=OUT/'data'/a.model
    lock=(folder/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():return
    if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker present')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    prep=json.loads((data/'preparation.json').read_text());graph=dict(np.load(data/'graph.npz'));feature=np.load(data/'features.npz')['x'];x=torch.from_numpy(feature);pairs=torch.from_numpy(graph['pairs']);attrs=torch.from_numpy(graph['attrs'])
    sets={s:dict(np.load(data/(s+'.npz'))) for s in ['train','val','test']};tensors={s:{k:torch.from_numpy(v) for k,v in d.items() if k in ['node','target','pixel_features']} for s,d in sets.items()}
    counts=prep['class_counts']['train'];weights=np.clip(np.sqrt(max(counts)/np.maximum(counts,1)),1,5)*np.array([1,1.4,1.8]);weights=torch.tensor(weights,dtype=torch.float32)
    train_pair=np.flatnonzero(graph['pair_split']==0);val_pair=np.flatnonzero(graph['pair_split']==1);test_pair=np.flatnonzero(graph['pair_split']==2)
    assert all(len(p)>0 for p in [train_pair,val_pair,test_pair])
    conn_y=torch.tensor(graph['pair_target'].astype(np.float32));cc=np.bincount(graph['pair_target'][train_pair],minlength=2);posweight=torch.tensor(float(np.clip(cc[0]/max(cc[1],1),.25,8)))
    # Derive class-boundary targets solely from pure temporal labels in the TRAIN spatial mask.
    hist=np.zeros((len(x),3),np.int64);np.add.at(hist,(sets['train']['node'],sets['train']['target']),1)
    pure=(hist.sum(1)>=3)&(hist.max(1)/np.maximum(hist.sum(1),1)>=.9);dom=hist.argmax(1);u,v=graph['pairs'].T
    boundary_pairs=np.flatnonzero(pure[u]&pure[v]&(graph['node_split'][u]==0)&(graph['node_split'][v]==0)&(graph['phase'][u]==graph['phase'][v]))
    boundary_target=torch.tensor((dom[u[boundary_pairs]]!=dom[v[boundary_pairs]]).astype(np.float32))
    model=Model(x.shape[1],a.kind);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,eta_min=1e-5)
    config=dict(model=a.model,kind=a.kind,neural_source_SHA256=sha(Path(__file__).with_name('model.py')), epochs=100,seed=SEED,base_checkpoint_sha256=prep['base_checkpoint_sha256'],new_initialization=True,old_GAT_weights_loaded=False,temporal_targets='upload18',connection_targets='upload17 user_defined142 IDs only; target-only, no ID/color/parent input',binary_geometry_frozen=True,classes=['persistent','newly_observed','tip_extension'],class_weights=weights.tolist(),sampling='up to8 pixels per node per class, training spatial mask only',objective='class-weighted focal CE gamma1.5 +0.3 weighted classDice +0.35 true-ID linkBCE +0.1 class-boundaryBCE +0.003 residual penalty',selection='Validation only: .15 persistent Dice +.40 new Dice +.45 tip Dice +.05 connectionF1; never use test for decisions',evaluation='Same-field inner spatial masks, transductive RGB/prediction graph; not external root/tube generalization',optimizer='AdamW .001 cosine to .00001',parameter_count=sum(p.numel() for p in model.parameters()))
    if (folder/'config.json').exists():assert json.loads((folder/'config.json').read_text())==config
    else:js(folder/'config.json',config)
    def checkpoint(epoch):return dict(state_dict=model.state_dict(),config=config,epoch=epoch)
    first=1;history=[];best=-1.;elapsed0=0.;tic=time.monotonic()
    if (folder/'resume.pt').exists():
        c=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);assert c['config']==config;model.load_state_dict(c['state_dict']);optimizer.load_state_dict(c['optimizer']);schedule.load_state_dict(c['scheduler']);torch.set_rng_state(c['torch_rng']);first=c['epoch']+1;history=c['history'];best=c['best'];elapsed0=c['elapsed_seconds']
    else:save(folder/'initial.pt',checkpoint(0))
    def evaluate(h,connection,split,threshold=.5):
        d=tensors[split];pred=[]
        for lo in range(0,len(d['node']),16384):pred.append(model.logits(h,d['node'][lo:lo+16384],d['pixel_features'][lo:lo+16384]).argmax(1).numpy())
        m=measures(sets[split]['target'],np.concatenate(pred));ix=val_pair if split=='val' else test_pair
        return m,links(graph['pair_target'][ix],connection[ix].sigmoid().numpy(),threshold)
    model.eval()
    with torch.inference_mode():h,co,bo=model(x,pairs,attrs);baseline,_=evaluate(h,co,'val')
    if not (folder/'epoch0_metrics.json').exists():js(folder/'epoch0_metrics.json',baseline)
    for epoch in range(first,101):
        if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('Stopped; last complete epoch is saved')
        te=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);h,connection,boundary=model(x,pairs,attrs);tr=tensors['train'];z=model.logits(h,tr['node'],tr['pixel_features']);y=tr['target'];ce=F.cross_entropy(z,y,reduction='none');correct=z.softmax(1).gather(1,y[:,None])[:,0];focal=(weights[y]*(1-correct).pow(1.5)*ce).sum()/weights[y].sum()
        p=z.softmax(1);g=F.one_hot(y,3).float();dice=1-(2*(p*g).sum(0)+1)/(p.sum(0)+g.sum(0)+1);dice=(dice*torch.tensor([.15,.4,.45])).sum()
        conloss=F.binary_cross_entropy_with_logits(connection[train_pair],conn_y[train_pair],pos_weight=posweight)
        bl=F.binary_cross_entropy_with_logits(boundary[boundary_pairs],boundary_target) if len(boundary_pairs) else boundary.sum()*0
        reg=(z-tr['pixel_features'][:,:3]).square().mean();loss=focal+.3*dice+.35*conloss+.1*bl+.003*reg
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();lr=optimizer.param_groups[0]['lr'];schedule.step();model.eval()
        with torch.inference_mode():h,co,bo=model(x,pairs,attrs);vm,vl=evaluate(h,co,'val')
        ds=[v['Dice'] for v in vm['classes']];score=.15*ds[0]+.4*ds[1]+.45*ds[2]+.05*vl['F1']
        rec=dict(epoch=epoch,training_loss=float(loss.detach()),training_focal=float(focal.detach()),training_connection_BCE=float(conloss.detach()),validation_persistent_Dice=ds[0],validation_new_Dice=ds[1],validation_extension_Dice=ds[2],validation_macro_Dice=vm['macro_Dice'],validation_connection_F1=vl['F1'],validation_score=score,learning_rate=lr,seconds=time.monotonic()-te);history.append(rec)
        if score>best:best=score;save(folder/'best.pt',dict(**checkpoint(epoch),validation_temporal=vm,validation_connection=vl,score=score))
        elapsed=elapsed0+time.monotonic()-tic
        save(folder/'resume.pt',dict(**checkpoint(epoch),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),torch_rng=torch.get_rng_state(),history=history,best=best,elapsed_seconds=elapsed))
        csvout(folder/'history.csv',history);js(folder/'status.json',dict(stage='training',epoch_completed=epoch,epochs=100,best_score=best,pid=os.getpid(),last_epoch_seconds=rec['seconds'],elapsed_seconds=elapsed))
        if epoch==1 or epoch%10==0:print('EPOCH',a.model,a.kind,epoch,'Dice',ds,'linkF1',vl['F1'],'sec',rec['seconds'],flush=True)
    c=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(c['state_dict']);model.eval()
    # Set link threshold using validation only, then evaluate held-out targets once.
    with torch.inference_mode():
        h,co,bo=model(x,pairs,attrs);yp=co[val_pair].sigmoid().numpy();yy=graph['pair_target'][val_pair];thresholds=np.linspace(.1,.9,33);threshold=float(max(thresholds,key=lambda t:links(yy,yp,t)['F1']));test,testlink=evaluate(h,co,'test',threshold)
        baseline_test=measures(sets['test']['target'],sets['test']['pixel_features'][:,:3].argmax(1))
    np.savez_compressed(folder/'graph_inference.npz',embedding=h.numpy(),connection_probability=co.sigmoid().numpy(),class_boundary_probability=bo.sigmoid().numpy())
    js(folder/'test_metrics.json',dict(temporal=test,connection=testlink,baseline_temporal_same_pixels=baseline_test))
    # Check source identities and frozen root inputs again before declaring completion.
    geometry=json.loads((data/'geometry_complete.json').read_text())
    for item in geometry['immutable_inputs']+prep['immutable_baselines']:assert sha(item['path'])==item['sha256']
    js(folder/'completion.json',dict(state='complete',epochs=100,selected_epoch=c['epoch'],test=test,test_connection=testlink,baseline_test=baseline_test,checkpoint_sha256=sha(folder/'best.pt'),binary_geometry_unchanged=True,elapsed_seconds=elapsed0+time.monotonic()-tic))
    js(folder/'status.json',dict(stage='complete',epoch_completed=100,selected_epoch=c['epoch']))
    print('COMPLETE',a.model,a.kind,'selected',c['epoch'],flush=True)

if __name__=='__main__':main()
