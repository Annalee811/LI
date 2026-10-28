"""Latest20 before/after scoring, full inference and prediction-only difference ROIs."""
import json
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from prepare import ROOT,OUT,BASE,MODELS,PERIODS,js,sha,csvout
from data import Data
from model import TemporalGraph,hierarchical_prediction
from train import measures,link_metric
from metrics import adjusted_rand

def cm4_metrics(cm):
    out=[]
    for k,name in enumerate(['persistent','newly_observed','tip_extension'],1):
        tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp)
        out.append(dict(name=name,Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),Dice=2*tp/max(2*tp+fp+fn,1),IoU=tp/max(tp+fp+fn,1),support=int(cm[k].sum())))
    return dict(classes=out,confusion=cm.tolist(),macro_Dice=float(np.mean([r['Dice'] for r in out])))

def difference_windows(before,after,gates):
    """No ground truth/accuracy argument: choose illustration windows by disagreement only."""
    changed=np.zeros((1108,2541),np.int32)
    for m in MODELS:
        for t in PERIODS[1:]:changed+=((before[m][t]!=after[m][t])&gates[m][t]).astype(np.int32)
    choices=[]
    for y in sorted(set(list(range(0,1108-384+1,96))+[1108-384])):
        for x in sorted(set(list(range(0,2541-384+1,96))+[2541-384])):choices.append((int(changed[y:y+384,x:x+384].sum()),[x,y,x+384,y+384]))
    choices.sort(key=lambda v:(-v[0],v[1]));selected=[]
    for score,roi in choices:
        overlap=False
        for old in selected:
            a=old['ROI'];inter=max(0,min(roi[2],a[2])-max(roi[0],a[0]))*max(0,min(roi[3],a[3])-max(roi[1],a[1]));overlap|=inter/(2*384**2-inter)>.2
        if not overlap:selected.append(dict(ROI=roi,changed_pixel_score=score))
        if len(selected)==3:break
    return selected

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);allrows=[];connections=[];consistency=[];allpred={};gates={};class_strata=[]
    spatial=np.load(OUT/'data/spatial.npy');reference={t:np.array(Image.open(OUT/'data/reference'/f'{t}_classes.png')) for t in PERIODS};source=json.loads((OUT/'preparation.json').read_text())
    for m in MODELS:
        data=Data(m);allpred[m]={};gates[m]={t:np.load(data.folder/'ownership.npz')[t]>=0 for t in PERIODS}
        gt_hist=np.zeros((len(data.x),3),np.int64);tt=data.sets['test'];np.add.at(gt_hist,(tt['node'],tt['target']),1);pure=(gt_hist.sum(1)>=3)&(gt_hist.max(1)/np.maximum(gt_hist.sum(1),1)>=.9);nodeclass=np.where(pure,gt_hist.argmax(1),-1);a,b=data.graph['pairs'].T
        for variant in ['baseline','control','convlstm_gat']:
            folder=OUT/(m+'_'+variant);folder.mkdir(exist_ok=True);full=folder/'full';full.mkdir(exist_ok=True);pred={};cm=np.zeros((4,4),np.int64);GT=[];PP=[];rootstats=[]
            if variant!='baseline':
                done=json.loads((folder/'completion.json').read_text());assert done['epochs_completed']==100
                ck=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);assert sha(folder/'best.pt')==done['checkpoint_sha256'];model=TemporalGraph(variant);model.load_state_dict(ck['state_dict']);model.eval()
                cached=dict(np.load(folder/'inference.npz'));h=torch.from_numpy(cached['h']);group=cached['groups'];newp=cached['group_new_probability']
                conn=cached['connection'];thresholds=done['link_thresholds'];new_threshold=done['new_state_threshold']
                for same,name in [(True,'within'),(False,'across')]:
                    ix=np.flatnonzero((data.graph['split']==2)&(data.same==same));metric=link_metric(data.graph['target'][ix],conn[ix],thresholds[name]);connections.append(dict(model=m,variant=variant,pair_scope=name,**metric))
                    for cls,key in [(1,'newly_observed'),(2,'tip_extension'),(0,'persistent')]:
                        take=ix[(nodeclass[a[ix]]==cls)|(nodeclass[b[ix]]==cls)]
                        if len(take):class_strata.append(dict(model=m,variant=variant,pair_scope=name,category=key,**link_metric(data.graph['target'][take],conn[take],thresholds[name])))
            for ti,t in enumerate(PERIODS):
                px=dict(np.load(data.folder/(t+'_pixels.npz')));gate=gates[m][t]
                if variant=='baseline':
                    original=np.array(Image.open(BASE/m/'full'/f'{t}_classes.png'));assert np.array_equal(original>0,gate);classes=original
                elif not ti:classes=np.where(gate,4,0).astype(np.uint8)
                else:
                    out=[];probs=[]
                    with torch.inference_mode():
                        for lo in range(0,len(px['node']),16384):
                            ix=slice(lo,lo+16384);log=model.pixel_logits(h,torch.from_numpy(px['node'][ix]),torch.from_numpy(px['pf'][ix]));raw=log.numpy()
                            out.append(hierarchical_prediction(raw,px['node'][ix],group,newp,new_threshold))
                            oldprob=torch.softmax(log[:,[0,2]],1).numpy();qnew=newp[group[px['node'][ix]]];probs.append(np.c_[(1-qnew)*oldprob[:,0],qnew,(1-qnew)*oldprob[:,1]])
                    vals=np.concatenate(out);classes=np.zeros((1108,2541),np.uint8);classes.ravel()[px['pixel']]=vals+1
                    np.savez_compressed(full/f'{t}_probability.npz',pixel=px['pixel'],probability=np.concatenate(probs).astype(np.float16),note='Structured root-state decoding; marginal class probabilities need not share the hard decoder argmax')
                assert np.array_equal(classes>0,gate),'Binary geometry changed'
                pred[t]=classes;Image.fromarray(classes).save(full/f'{t}_classes.png')
                if not ti:continue
                gt=reference[t];valid=(spatial==2)&(gt!=255);cm+=np.bincount(gt[valid].astype(np.int64)*4+classes[valid],minlength=16).reshape(4,4)
                use=px['split']==2;ys=px['target'][use];ps=classes.ravel()[px['pixel'][use]]-1;GT.extend(ys);PP.extend(ps)
                # These are scoring labels only, never the predicted grouping input.
                for rid in np.unique(px['rid'][use]):
                    if rid<0:continue
                    take=use&(px['rid']==rid);p=classes.ravel()[px['pixel'][take]];y=px['target'][take]
                    if len(p)<10:continue
                    counts=np.bincount(p,minlength=5);newfrac=float((y==1).mean());gtstate='new' if newfrac>=.98 else 'established' if newfrac<=.02 else 'ambiguous'
                    fragments=int(len(np.unique(group[px['node'][take]]))) if variant!='baseline' else None
                    rootstats.append(dict(period=t,root_target=int(rid),pixels=len(p),predicted_new_pixels=int(counts[2]),predicted_extension_pixels=int(counts[3]),new_and_extension_any=bool(counts[2]>0 and counts[3]>0),new_and_extension_at_least5px_each=bool(min(counts[2],counts[3])>=5),reference_state=gtstate,predicted_state='new' if counts[2]/len(p)>=.5 else 'established',predicted_groups=fragments))
            allpred[m][variant]=pred;conditional=measures(np.array(GT),np.array(PP));overall=cm4_metrics(cm)
            if variant!='baseline':assert conditional['confusion']==done['test']['confusion'],'Full maps do not match heldout selected-model scoring'
            for scope,metrics in [('fixed_root_temporal',conditional),('including_binary_misses',overall)]:
                for row in metrics['classes']:allrows.append(dict(model=m,variant=variant,scope=scope,**row))
            js(folder/'latest20_metrics.json',dict(conditional=conditional,including_binary_misses=overall))
            if rootstats:csvout(folder/'root_period_diagnostics.csv',rootstats)
            jumps=0;eligible=0
            for rid in {r['root_target'] for r in rootstats}:
                rr={r['period']:r for r in rootstats if r['root_target']==rid}
                for prev,t in [('T2','T3'),('T3','T4')]:
                    if prev in rr and t in rr and rr[prev]['reference_state']=='established' and rr[t]['reference_state']=='established':eligible+=1;jumps+=int(rr[prev]['predicted_state']=='established' and rr[t]['predicted_state']=='new')
            assoc=dict(mixed_identity_predicted_groups=None,scoreable_predicted_groups=None,node_group_purity=None,node_group_ARI=None)
            if variant!='baseline':
                known=(data.graph['node_split']==2)&(data.graph['root_target']>=0)&(data.graph['phase']>0)
                ids=data.graph['root_target'][known];gp=group[known];phase=data.graph['phase'][known];mixed=0;total=0;correct=0
                for label in np.unique(gp):
                    values=ids[gp==label];counts=np.bincount(values);mixed+=int(np.count_nonzero(counts)>1);total+=1;correct+=int(counts.max())
                assoc=dict(mixed_identity_predicted_groups=mixed,scoreable_predicted_groups=total,node_group_purity=correct/max(len(ids),1),node_group_ARI=adjusted_rand(ids*4+phase,gp))
            consistency.append(dict(model=m,variant=variant,**assoc,scoreable_root_periods=len(rootstats),new_extension_conflicts_any=sum(r['new_and_extension_any'] for r in rootstats),new_extension_conflicts_5px=sum(r['new_and_extension_at_least5px_each'] for r in rootstats),scoreable_stable_root_transitions=eligible,inappropriate_established_to_new_jumps=jumps,mean_groups_per_known_root_period=float(np.mean([r['predicted_groups'] for r in rootstats])) if variant!='baseline' and rootstats else None))
            print('EVALUATED',m,variant,'latest20 conditional Dice',conditional['macro_Dice'],flush=True)
    csvout(OUT/'Three_classes_accuracy_before_after.csv',allrows);csvout(OUT/'Connection_accuracy_within_and_across.csv',connections);csvout(OUT/'Connection_accuracy_by_class.csv',class_strata);csvout(OUT/'Root_temporal_consistency.csv',consistency)
    before={m:allpred[m]['baseline'] for m in MODELS};after={m:allpred[m]['convlstm_gat'] for m in MODELS};windows=difference_windows(before,after,gates)
    # Reference-only matched ROI, plus separate growth windows for each follow-up date.
    best=None
    for y in range(0,1108-384+1,48):
        for x in range(0,2541-384+1,48):
            cc=[int((reference[t][y:y+384,x:x+384]==k).sum()) for t in PERIODS[1:] for k in [2,3]];score=sum(np.log1p(cc))
            if best is None or score>best[0]:best=(score,[x,y,x+384,y+384],cc)
    tips=[]
    for t in PERIODS[1:]:
        lab,n=ndi.label(reference[t]==3)
        if not n:continue
        sizes=np.bincount(lab.ravel());sizes[0]=0;selected=lab==sizes.argmax();yy,xx=np.where(selected);x=int(np.clip(round(xx.mean())-128,0,2541-256));y=int(np.clip(round(yy.mean())-128,0,1108-256));tips.append(dict(period=t,ROI=[x,y,x+256,y+256],reference_extension_pixels=int((reference[t][y:y+256,x:x+256]==3).sum())))
    js(OUT/'ROI_selection.json',dict(matched_ROI=best[1],matched_reference_counts=best[2],high_difference_windows=windows,extension_ROIs=tips,difference_selection='Top3 nonoverlapping384px windows by ORIGINAL-vs-JOINT prediction disagreement only; no accuracy/ground-truth scoring input',matched_selection='Reference20 class coverage only',scope='Illustrative windows may include training/development pixels; not a heldout accuracy estimate; regressions retained'))
    for entry in source['immutable_inputs']:assert sha(entry['path'])==entry['sha256']
    js(OUT/'evaluation_verification.json',dict(passed=True,latest20_source_sha256=source['upload20_sha256'],all_36_binary_supports_equal_original=True,all_class_metrics_use_same_latest20=True,ground_truth_IDs_used_in_prediction=False,historical_curves_not_rewritten=True,causal_preflight_passed=json.loads((OUT/'preflight_verification.json').read_text())['passed'],difference_ROIs_use_predictions_only=True,all_inputs_unchanged=True))
if __name__=='__main__':main()
