"""Fixed-shape full inference and honest class/connection/continuity comparisons."""
import json,sys
import numpy as np
from PIL import Image
import torch
from model import Model
from prepare import OUT,BASE,ROOT,MODELS,PERIODS,js,csvout,sha,rows
from train import measures

def continuity(graph,identity,prediction,target,node_counts,split):
    a,b=graph['pairs'].T
    # Only known same roots within one phase and the heldout spatial mask.
    pair=(identity[a]>=0)&(identity[a]==identity[b])&(graph['phase'][a]==graph['phase'][b])&(graph['phase'][a]>0)&(graph['node_split'][a]==split)&(graph['node_split'][b]==split)
    pure=(node_counts.sum(1)>=3)&(node_counts.max(1)/np.maximum(node_counts.sum(1),1)>=.9)
    pair &= pure[a]&pure[b];same=pair&(target[a]==target[b]);change=pair&(target[a]!=target[b])
    return dict(reference_constant_same_root_pairs=int(same.sum()),unnecessary_class_switches=int((prediction[a[same]]!=prediction[b[same]]).sum()),unnecessary_class_switch_rate=float((prediction[a[same]]!=prediction[b[same]]).mean()) if same.any() else None,
        reference_true_class_transition_pairs=int(change.sum()),true_transition_preservation_rate=float((prediction[a[change]]!=prediction[b[change]]).mean()) if change.any() else None,
        scope='Candidate-nearby same-root pairs with pure heldout temporal labels; valid class changes are scored separately, not penalized')

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);allmetrics=[];alllinks=[];cont=[]
    spatial=np.full((1108,2541),-1,np.int8)
    for r in rows(BASE/'data/spatial_manifest.csv'):
        x,y,w,h=[int(r[k]) for k in ['x','y','width','height']];spatial[y:y+h,x:x+w]=['train','val','test'].index(r['split'])
    for m in MODELS:
        data=OUT/'data'/m;prep=json.loads((data/'preparation.json').read_text());graph=dict(np.load(data/'graph.npz'));identity=np.load(data/'identity_targets.npz')['root_target'];N=len(graph['features'])
        test=dict(np.load(data/'test.npz'));hist=np.zeros((N,3),np.int64);np.add.at(hist,(test['node'],test['target']),1);true=hist.argmax(1)
        for kind in ['baseline','mlp','gat']:
            folder=OUT/(m+'_'+kind);folder.mkdir(exist_ok=True);full=folder/'full';full.mkdir(exist_ok=True)
            if kind!='baseline':
                completion=json.loads((folder/'completion.json').read_text());assert completion['epochs']==100
                checkpoint=torch.load(folder/'best.pt',weights_only=False,map_location='cpu');assert sha(folder/'best.pt')==completion['checkpoint_sha256']
                model=Model(prep['input_features'],kind);model.load_state_dict(checkpoint['state_dict']);model.eval();h=torch.from_numpy(np.load(folder/'graph_inference.npz')['embedding'])
                link=completion['test_connection'];alllinks.append(dict(model=m,variant=kind,**link))
                scores=np.load(folder/'graph_inference.npz')['connection_probability'];edge_records=[]
                for ix in np.flatnonzero(graph['pair_split']==2):
                    aa,bb=graph['pairs'][ix];pa=graph['phase'][aa];pb=graph['phase'][bb]
                    edge_records.append(dict(node_a=int(aa),node_b=int(bb),period_a=PERIODS[pa],period_b=PERIODS[pb],x_a=float(graph['centers'][aa,0]),y_a=float(graph['centers'][aa,1]),x_b=float(graph['centers'][bb,0]),y_b=float(graph['centers'][bb,1]),reference_same_root=int(graph['pair_target'][ix]),predicted_same_root_probability=float(scores[ix]),threshold=link['threshold'],predicted_connection=bool(scores[ix]>=link['threshold'])))
                csvout(folder/'Heldout_connection_predictions.csv',edge_records)
            pixelpred=[];nodepredhist=np.zeros((N,3),np.int64);cm=np.zeros((4,4),np.int64)
            for t in PERIODS:
                px=dict(np.load(data/(t+'_pixels.npz')));owner=np.load(data/'ownership.npz')[t];gate=owner>=0
                if t=='T1':pred=np.where(gate,4,0).astype(np.uint8)
                else:
                    q=np.load(str(prep['baseline_full'])+'/'+t+'_probability.npz')['probability'].astype(np.float32);q/=np.maximum(q.sum(0),1e-8)
                    base=q.reshape(3,-1).T[px['pixel']];pf=np.c_[np.log(np.maximum(base,1e-5)),px['xy_offset']].astype(np.float32)
                    if kind=='baseline':score=base
                    else:
                        vals=[]
                        with torch.inference_mode():
                            for lo in range(0,len(pf),16384):vals.append(model.logits(h,torch.from_numpy(px['node'][lo:lo+16384]),torch.from_numpy(pf[lo:lo+16384])).softmax(1).numpy())
                        score=np.concatenate(vals)
                    out=score.argmax(1);pred=np.zeros((1108,2541),np.uint8);pred.ravel()[px['pixel']]=out+1
                    selected=px['split']==2;np.add.at(nodepredhist,(px['node'][selected],out[selected]),1);pixelpred.extend(out[selected].tolist())
                    gt=np.array(Image.open(BASE/'data/current'/f'{t}_classes.png'));valid=(spatial==2)&(gt!=255)
                    cm+=np.bincount(gt[valid].astype(np.int64)*4+pred[valid],minlength=16).reshape(4,4)
                    np.savez_compressed(full/f'{t}_root_probabilities.npz',pixel=px['pixel'],probability=score.astype(np.float16))
                assert np.array_equal(pred>0,gate)
                Image.fromarray(pred).save(full/f'{t}_classes.png')
            # Compute conditional scores in the exact saved test-sample order.
            yy=[];pp=[]
            for ti,t in enumerate(PERIODS[1:],1):
                sel=test['period']==ti;yy.extend(test['target'][sel]);pred=np.array(Image.open(full/f'{t}_classes.png'));pp.extend(pred.ravel()[test['pixel'][sel]]-1)
            metric=measures(np.array(yy),np.array(pp))
            expected=json.loads((OUT/(m+'_gat')/'completion.json').read_text())['baseline_test'] if kind=='baseline' else completion['test']
            assert metric['confusion']==expected['confusion'], 'Full-map predictions disagree with selected-head heldout evaluation'
            js(folder/'full_conditional_metrics.json',metric)
            for row in metric['classes']:allmetrics.append(dict(model=m,variant=kind,scope='Fixed binary roots, inner spatial holdout',**row))
            end=[]
            for k in [1,2,3]:
                tp=int(cm[k,k]);fn=int(cm[k].sum()-tp);fp=int(cm[:,k].sum()-tp)
                end.append(dict(name=['persistent','newly_observed','tip_extension'][k-1],Dice=2*tp/max(2*tp+fp+fn,1),Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),IoU=tp/max(tp+fp+fn,1)))
            js(folder/'end_to_end_metrics.json',dict(classes=end,confusion=cm.tolist(),scope='Explicitly annotated pixels; frozen binary misses included'))
            cr=continuity(graph,identity,nodepredhist.argmax(1),true,hist,2);cont.append(dict(model=m,variant=kind,**cr))
            print('EVALUATED',m,kind,'conditional Dice',metric['macro_Dice'],flush=True)
    csvout(OUT/'Temporal_class_metrics_before_after.csv',allmetrics);csvout(OUT/'Connection_metrics_GAT_control.csv',alllinks);csvout(OUT/'Class_continuity_comparison.csv',cont)
    # Old and new losses cannot be conflated; preserve every saved epoch and source hash.
    for m in MODELS:
        geo=json.loads((OUT/'data'/m/'geometry_complete.json').read_text());prep=json.loads((OUT/'data'/m/'preparation.json').read_text())
        for item in geo['immutable_inputs']+prep['immutable_baselines']:assert sha(item['path'])==item['sha256']
        for kind in ['gat','mlp']:
            history=rows(OUT/(m+'_'+kind)/'history.csv');assert [int(r['epoch']) for r in history]==list(range(1,101))
    js(OUT/'evaluation_verification.json',dict(passed=True,all_12_frozen_supports_unchanged=True,all_6_new_modules_100_actual_epochs=True,temporal_source='upload18 only',connection_source='upload17 user_defined142 IDs only',all_source_hashes_unchanged=True,primary_metric='Conditional temporal scores on fixed binary supports',scoring_scope='Same-field internal spatial holdout, not independent generalization'))

if __name__=='__main__':main()
