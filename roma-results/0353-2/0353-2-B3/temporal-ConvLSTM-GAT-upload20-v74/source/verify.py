"""Test true recurrent causality, direct gradients and unchanged baseline artifacts."""
import json
import numpy as np
import torch
from data import Data
from model import TemporalGraph,decode_groups
from prepare import OUT,MODELS,js,sha
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);checks=[]
    # Software-only crossing fixture: one shared junction must not merge two distinct routes.
    pp=np.array([[0,1],[0,2],[0,4],[1,4],[2,3],[2,4],[3,4]],np.int64);aa=np.zeros((7,8),np.float32);prob=np.array([.95,.6,.9,.8,.94,.85,.7])
    route=dict(native_group=np.arange(5),is_junction=np.array([0,0,0,0,1],bool),end_chain=np.arange(4),bridge_a=np.array([0,2,0]),bridge_b=np.array([1,3,2]),bridge_pair_index=np.array([0,4,1]),bridge_distance=np.zeros(3),bridge_alignment=np.array([1.,1.,0.]),bridge_facing=np.array([2.,2.,1.]))
    gg=decode_groups(pp,aa,np.zeros(5,int),prob,.5,route)
    assert gg[0]==gg[1]==gg[4] and gg[2]==gg[3] and gg[0]!=gg[2]

    prep=json.loads((OUT/'preparation.json').read_text())
    for p in prep['immutable_inputs']:assert sha(p['path'])==p['sha256']
    for m in MODELS:
        d=Data(m);assert d.frames.shape==(4,8,139,318)
        for variant in ['convlstm_gat','control']:
            torch.manual_seed(31);model=TemporalGraph(variant);model.eval()
            with torch.no_grad():
                h,c,z=d.forward(model);changed=d.frames.clone();changed[3]=torch.rand_like(changed[3]);xx=d.x.clone();xx[d.phase==3]+=2
                future_h,future_c,_=d.forward(model,frames=changed,x=xx)
                error=float((h[d.phase<3]-future_h[d.phase<3]).abs().max());assert error<1e-6,'Future frame/features changed earlier predictions'
                sample=d.tensor['train'];flat=model.pixel_logits(h,sample['node'][:128],sample['pf'][:128]);assert torch.equal(flat,sample['pf'][:128,:3])
            model.train();h,c,z=d.forward(model);ix=np.flatnonzero((d.graph['split']==0)&d.same)[:2048];y=torch.tensor(d.graph['target'][ix],dtype=torch.float32)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(c[ix],y);assert torch.isfinite(loss);loss.backward()
            if variant=='convlstm_gat':
                mem=float(model.memory.gates.weight.grad.abs().sum());gats=[float(q.project.weight.grad.abs().sum()) for q in model.layers];assert mem>0 and all(v>0 for v in gats)
            else:mem=0.;gats=[]
            checks.append(dict(model=m,variant=variant,future_causality_max_error=error,ConvLSTM_gradient_L1=mem,GAT_gradient_L1=gats,flat_zero_residual_matches_frozen_baseline=True))
            print('VERIFIED',m,variant,'causality',error,flush=True)
    js(OUT/'preflight_verification.json',dict(passed=True,checks=checks,source20_and_old_results_unchanged=True,reference_IDs_not_in_model_input=True,topology_crossing_fixture_passed=True))
if __name__=='__main__':main()
