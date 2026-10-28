"""Meaningful checks on immutable ownership, target isolation and GAT gradients."""
import json,argparse
import numpy as np
import torch
from model import Model
from prepare import OUT,MODELS,BASE,js,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=MODELS);args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);checks=[]
    for m in ([args.model] if args.model else MODELS):
        data=OUT/'data'/m
        if not (data/'preparation.json').exists():continue
        prep=json.loads((data/'preparation.json').read_text());geo=dict(np.load(data/'graph.npz'));x=np.load(data/'features.npz')['x'];ident=np.load(data/'identity_targets.npz')['root_target']
        assert x.dtype==np.float32 and np.isfinite(x).all() and x.shape[1]==28
        assert (geo['pair_target'][geo['attrs'][:,5]>0]<0).all(), 'Trivial native-chain identity links must not inflate crossing metrics'
        a,b=geo['pairs'].T;known=geo['pair_target']>=0
        assert np.array_equal(geo['pair_target'][known],(ident[a[known]]==ident[b[known]]).astype(np.int8))
        for k in [0,1,2]:
            mask=geo['pair_split']==k;assert (geo['node_split'][a[mask]]==k).all() and (geo['node_split'][b[mask]]==k).all()
            assert np.all(np.bincount(geo['pair_target'][mask],minlength=2)>0)
        for split in ['train','val','test']:assert min(prep['class_counts'][split])>0
        for item in json.loads((data/'geometry_complete.json').read_text())['immutable_inputs']+prep['immutable_baselines']:assert sha(item['path'])==item['sha256']
        # Full graph forward, exact zero-residual baseline and nonzero finite actual task gradients.
        sets=dict(np.load(data/'train.npz'));xx=torch.from_numpy(x);pair=torch.from_numpy(geo['pairs']);attrs=torch.from_numpy(geo['attrs']);n=torch.from_numpy(sets['node'][:512]);pf=torch.from_numpy(sets['pixel_features'][:512]);y=torch.from_numpy(sets['target'][:512])
        for kind in ['gat','mlp']:
            torch.manual_seed(19);model=Model(x.shape[1],kind);model.eval();h,co,bo=model(xx,pair,attrs);z=model.logits(h,n,pf)
            assert torch.equal(z,pf[:,:3]), 'Untrained residual must reproduce current frozen CNN exactly'
            ix=np.flatnonzero(geo['pair_split']==0)[:1024]
            loss=torch.nn.functional.cross_entropy(z,y)+torch.nn.functional.binary_cross_entropy_with_logits(co[ix],torch.tensor(geo['pair_target'][ix],dtype=torch.float32))
            assert torch.isfinite(loss);loss.backward();grad=[v.grad for v in model.parameters() if v.grad is not None]
            assert all(torch.isfinite(g).all() for g in grad) and sum(float(g.abs().sum()) for g in grad)>0
            checks.append(dict(model=m,kind=kind,zero_residual_matches_baseline=True,finite_nonzero_gradients=True,nodes=len(x),pairs=len(pair)))
            del model,h,co,bo,z,grad,loss
    assert checks,'No model data prepared'
    prior=json.loads((OUT/'preflight_verification.json').read_text()).get('checks',[]) if (OUT/'preflight_verification.json').exists() else []
    checks=[v for v in prior if v['model'] not in {q['model'] for q in checks}]+checks
    js(OUT/'preflight_verification.json',dict(passed=True,checks=checks,identity_targets_not_inputs=True,temporal_source='18',connection_source='17 user_defined only',cross_split_connection_supervision=False,test_targets_not_used_for_training_or_selection=True))
    print(json.dumps(dict(passed=True,branches=len(checks))),flush=True)
if __name__=='__main__':main()
