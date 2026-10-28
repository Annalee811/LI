"""Real100-epoch causal history/graph training and matched updated-label controls."""
import argparse,fcntl,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import TemporalGraph,pool_groups,decode_groups,group_state,hierarchical_prediction
from data import Data
from prepare import OUT,MODELS,SEED,js,sha,csvout
from metrics import binary_curve

def save(p,v):tmp=p.with_suffix('.tmp');torch.save(v,tmp);os.replace(tmp,p)
def measures(y,p):
    cm=np.bincount(y*3+p,minlength=9).reshape(3,3);out=[]
    for k,name in enumerate(['persistent','newly_observed','tip_extension']):
        tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp)
        out.append(dict(name=name,Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),Dice=2*tp/max(2*tp+fp+fn,1),IoU=tp/max(tp+fp+fn,1),support=int(cm[k].sum())))
    return dict(classes=out,macro_Dice=float(np.mean([v['Dice'] for v in out])),confusion=cm.tolist())
def link_metric(y,p,threshold=.5):
    q=p>=threshold;tp=int((q&(y==1)).sum());fp=int((q&(y==0)).sum());fn=int((~q&(y==1)).sum());pr=tp/max(tp+fp,1);re=tp/max(tp+fn,1);curve=binary_curve(y==1,p)
    return dict(Precision=pr,Recall=re,F1=2*tp/max(2*tp+fp+fn,1),F05=1.25*pr*re/max(.25*pr+re,1e-10),AP=curve['average_precision'],AUC=curve['AUC'],threshold=float(threshold),positive_pairs=int((y==1).sum()),negative_pairs=int((y==0).sum()))
def evaluate(model,data,h,connection,split='val',link_threshold=.5,state_threshold=.5):
    group=decode_groups(data.graph['pairs'],data.graph['attrs'],data.graph['phase'],connection,link_threshold,routing=data.routing)
    state=group_state(model,h,group).sigmoid().cpu().numpy();d=data.tensor[split];flat=[]
    for lo in range(0,len(d['node']),16384):flat.append(model.pixel_logits(h,d['node'][lo:lo+16384],d['pf'][lo:lo+16384]).cpu().numpy())
    logits=np.concatenate(flat);pred=hierarchical_prediction(logits,data.sets[split]['node'],group,state,state_threshold)
    return measures(data.sets[split]['target'],pred),measures(data.sets[split]['target'],logits.argmax(1)),group,state,logits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS,required=True);ap.add_argument('--variant',choices=['control','convlstm_gat'],required=True);a=ap.parse_args()
    folder=OUT/(a.model+'_'+a.variant);folder.mkdir(exist_ok=True);lock=(folder/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'completion.json').exists():return
    if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('User stop marker present')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    data=Data(a.model);model=TemporalGraph(a.variant);tr=data.tensor['train'];hist=data.graph['train_class_hist'];counts=hist.sum(0);weights=torch.tensor(np.clip(np.sqrt(counts.max()/np.maximum(counts,1)),1,4),dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0003);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100,eta_min=1e-5)
    prep=json.loads((OUT/'preparation.json').read_text());config=dict(model=a.model,variant=a.variant,epochs=100,seed=SEED,upload20_sha256=prep['upload20_sha256'],random_new_modules=True,old_optimized_weights_loaded=False,baseline_probabilities_frozen=True,binary_support_fixed=True,causal_ConvLSTM=True,causal_GAT=True,ConvLSTM_enabled=(a.variant=='convlstm_gat'),GAT_enabled=(a.variant=='convlstm_gat'),loss_curves='Training and validation share the same weighted pixel CE; composite optimization objective separately logged',real_ConvLSTM_spatial_shape=[139,318],root_identity_target_only=True,root_state_learned=True,decoder='One-to-one endpoint continuation, single-owner junctions; predicted root groups and learned new/established state; persistent/tip within established group; no ground truth IDs at inference',objective='weighted pixel CE +.2 macro softDice +.25 root-state BCE +.3 balanced within/cross link BCE +.1 native continuity BCE +.05 stable-root temporal consistency +.003 residual penalty',selection='Validation hierarchical Dice: .2 persistent +.4 new +.4 tip, plus .05 within-period AP; epochs1..100',reference_scope='Latest20 model-assisted revised reference; one B3 development field, internal spatial holdout, not independent accuracy',graph_sha256=sha(data.folder/'graph.npz'),frames_sha256=sha(data.folder/'frames.npz'),model_source_sha256=sha(Path(__file__).with_name('model.py')),weights=weights.tolist())
    if (folder/'config.json').exists():assert json.loads((folder/'config.json').read_text())==config
    else:js(folder/'config.json',config)
    def ck(epoch):return dict(state_dict=model.state_dict(),config=config,epoch=epoch)
    first=1;history=[];best=-1.;elapsed0=0.;tic=time.monotonic()
    if (folder/'resume.pt').exists():
        c=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);assert c['config']==config;model.load_state_dict(c['state_dict']);optimizer.load_state_dict(c['optimizer']);schedule.load_state_dict(c['scheduler']);torch.set_rng_state(c['torch_rng']);history=c['history'];best=c['best'];elapsed0=c['elapsed_seconds'];first=c['epoch']+1
    else:
        save(folder/'initial.pt',ck(0));model.eval()
        with torch.inference_mode():
            h0,c0,z0=data.forward(model);train0=model.pixel_logits(h0,tr['node'],tr['pf']);vv=data.tensor['val'];val0=model.pixel_logits(h0,vv['node'],vv['pf'])
            m0,_,_,_,_=evaluate(model,data,h0,c0.sigmoid().numpy())
            js(folder/'epoch0_metrics.json',dict(epoch=0,training_monitoring_CE=float(F.cross_entropy(train0,tr['target'],weight=weights)),validation_monitoring_CE=float(F.cross_entropy(val0,vv['target'],weight=weights)),validation=m0))
        del h0,c0,z0,train0,val0
    phase_train={same:np.flatnonzero((data.graph['split']==0)&(data.same==same)) for same in [True,False]}
    valwithin=np.flatnonzero((data.graph['split']==1)&data.same);target=torch.tensor(data.graph['target'],dtype=torch.float32)
    rs=data.state;new_n=int((rs['target']==1).sum());old_n=int((rs['target']==0).sum());state_pos=torch.tensor(old_n/max(new_n,1),dtype=torch.float32)
    for epoch in range(first,101):
        if (OUT/'STOPPED_BY_USER.json').exists():raise SystemExit('Stopped; latest whole epoch is saved')
        te=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);h,connection,z=data.forward(model);logits=model.pixel_logits(h,tr['node'],tr['pf']);y=tr['target']
        ce=F.cross_entropy(logits,y,weight=weights,reduction='none');junction=tr['pf'][:,14];factor=1+junction*2*(y!=2).float();pixel_loss=(ce*factor).sum()/(weights[y]*factor).sum().clamp_min(1)
        p=logits.softmax(1);gt=F.one_hot(y,3).float();dice=1-((2*(p*gt).sum(0)+1)/(p.sum(0)+gt.sum(0)+1)).mean()
        state_logits=model.root_state(pool_groups(h,rs['nodes'],rs['ptr']))[:,0];state_loss=F.binary_cross_entropy_with_logits(state_logits,rs['target'].float(),pos_weight=state_pos)
        links=[]
        for same in [True,False]:
            ix=phase_train[same];cc=np.bincount(data.graph['target'][ix],minlength=2);pw=torch.tensor(float(cc[0]/max(cc[1],1)))
            links.append(F.binary_cross_entropy_with_logits(connection[ix],target[ix],pos_weight=pw))
        link_loss=torch.stack(links).mean()
        native=F.binary_cross_entropy_with_logits(connection[data.native_train],data.native_target) if len(data.native_train) else connection.sum()*0
        stable=logits.sum()*0
        if len(data.stable_pairs):
            count=torch.bincount(tr['node'],minlength=len(h));probs=logits.new_zeros((len(h),3));probs.index_add_(0,tr['node'],p);probs=probs/count.clamp_min(1)[:,None];ab=data.pairs[data.stable_pairs];valid=(count[ab[:,0]]>0)&(count[ab[:,1]]>0)
            if valid.any():stable=(probs[ab[valid,0]]-probs[ab[valid,1]]).square().mean()
        reg=(logits-tr['pf'][:,:3]).square().mean();loss=pixel_loss+.2*dice+.25*state_loss+.3*link_loss+.1*native+.05*stable+.003*reg
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
        if epoch==1 and a.variant=='convlstm_gat':
            checks=dict(memory_gradient=float(model.memory.gates.weight.grad.abs().sum()),GAT_gradients=[float(q.project.weight.grad.abs().sum()) for q in model.layers])
            assert checks['memory_gradient']>0 and all(v>0 for v in checks['GAT_gradients']);js(folder/'gradient_verification.json',dict(passed=True,**checks))
        optimizer.step();lr=optimizer.param_groups[0]['lr'];schedule.step();model.eval()
        with torch.inference_mode():
            vh,vc,vz=data.forward(model);prob=vc.sigmoid().numpy();vm,raw,_,_,_=evaluate(model,data,vh,prob);vl=link_metric(data.graph['target'][valwithin],prob[valwithin])
            # Validation objective uses the same training losses' class task, not test labels.
            vv=data.tensor['val'];vlog=model.pixel_logits(vh,vv['node'],vv['pf']);vloss=float(F.cross_entropy(vlog,vv['target'],weight=weights))
        ds=[r['Dice'] for r in vm['classes']];score=.2*ds[0]+.4*ds[1]+.4*ds[2]+.05*vl['AP']
        rec=dict(epoch=epoch,training_loss=float(loss.detach()),training_pixel_CE=float(pixel_loss.detach()),training_monitoring_CE=float(F.cross_entropy(logits.detach(),y,weight=weights)),training_root_state_BCE=float(state_loss.detach()),training_connection_BCE=float(link_loss.detach()),validation_pixel_CE=vloss,validation_hierarchical_macro_Dice=vm['macro_Dice'],validation_persistent_Dice=ds[0],validation_new_Dice=ds[1],validation_extension_Dice=ds[2],validation_within_connection_AP=vl['AP'],validation_score=score,learning_rate=lr,seconds=time.monotonic()-te);history.append(rec)
        if score>best:best=score;save(folder/'best.pt',dict(**ck(epoch),validation=vm,within_connection=vl,score=score))
        elapsed=elapsed0+time.monotonic()-tic;save(folder/'resume.pt',dict(**ck(epoch),optimizer=optimizer.state_dict(),scheduler=schedule.state_dict(),torch_rng=torch.get_rng_state(),history=history,best=best,elapsed_seconds=elapsed));csvout(folder/'history.csv',history);js(folder/'status.json',dict(stage='training',epoch_completed=epoch,epochs_total=100,pid=os.getpid(),last_epoch_seconds=rec['seconds'],elapsed_seconds=elapsed,best_validation_score=best))
        if epoch==1 or epoch%10==0:print('EPOCH',a.model,a.variant,epoch,'Dice',ds,'withinAP',vl['AP'],'seconds',rec['seconds'],flush=True)
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['state_dict']);model.eval()
    with torch.inference_mode():
        h,connection,z=data.forward(model);prob=connection.sigmoid().numpy();thresholds={}
        for same,name in [(True,'within'),(False,'across')]:
            ix=np.flatnonzero((data.graph['split']==1)&(data.same==same));thresholds[name]=float(max(np.linspace(.1,.9,33),key=lambda t:link_metric(data.graph['target'][ix],prob[ix],t)['F05']))
        best_state=None
        for st in [.35,.5,.65]:
            val,_,_,_,_=evaluate(model,data,h,prob,link_threshold=thresholds['within'],state_threshold=st);ds=[r['Dice'] for r in val['classes']];score=.2*ds[0]+.4*ds[1]+.4*ds[2]
            if best_state is None or score>best_state[0]:best_state=(score,st)
        test,raw,groups,state,_=evaluate(model,data,h,prob,'test',thresholds['within'],best_state[1])
    testlinks={}
    for same,name in [(True,'within'),(False,'across')]:
        ix=np.flatnonzero((data.graph['split']==2)&(data.same==same));testlinks[name]=link_metric(data.graph['target'][ix],prob[ix],thresholds[name])
    np.savez_compressed(folder/'inference.npz',h=h.numpy(),connection=prob,embedding=z.numpy(),groups=groups,group_new_probability=state)
    js(folder/'test_metrics.json',dict(hierarchical=test,flat_auxiliary=raw,connection=testlinks,new_state_threshold=best_state[1],link_thresholds=thresholds))
    for entry in prep['immutable_inputs']:assert sha(entry['path'])==entry['sha256']
    js(folder/'completion.json',dict(state='complete',epochs_completed=100,selected_epoch=selected['epoch'],test=test,raw_auxiliary=raw,connection=testlinks,new_state_threshold=best_state[1],link_thresholds=thresholds,checkpoint_sha256=sha(folder/'best.pt'),elapsed_seconds=elapsed0+time.monotonic()-tic))
    js(folder/'status.json',dict(stage='complete',epoch_completed=100,selected_epoch=selected['epoch']));print('COMPLETE',a.model,a.variant,flush=True)
if __name__=='__main__':main()
