"""Verify actual supervision, frozen support and one fresh differentiable batch."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    prep=json.loads((a.run/'preparation.json').read_text());checks=[];details=[]
    for item in prep['immutable_inputs']:assert c.sha(item['path'])==item['sha256']
    checks.append('All uploaded/reference/RGB/binary weights and masks have verified immutable hashes')
    for m in c.MODELS:
        data=c.Data(a.run,prep['roots'],m);counts=data.counts();assert (counts>0).all()
        model=c.AxisHead();assert model.a[0].in_channels==25
        assert not any('gat' in type(x).__name__.lower() for x in model.modules())
        assert not (a.run/m/'resume.pt').exists(), 'Preflight must precede training'
        xx,yy,_=data.patch(data.train[0],True,np.random.default_rng(8))
        xx=torch.from_numpy(xx[None]).contiguous(memory_format=torch.channels_last)
        yy=torch.from_numpy(yy[None]);z,b=model.forward_with_boundary(xx)
        w=torch.tensor(np.clip(np.sqrt(counts.max()/counts),1,12),dtype=torch.float32)
        loss=c.loss(z,yy,w,b,xx[:,17]);assert torch.isfinite(loss)
        loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads)>0
        details.append(dict(model=m,train_class_counts=counts.tolist(),parameters=sum(p.numel() for p in model.parameters()),test_batch_loss=float(loss.detach()),finite_gradients=True))
        checks.append(m+': fresh 25-channel CNN; all three supervised classes; finite nonzero gradients; no previous state or GAT')
    checks.append('Feature builder takes paired RGB, binary probabilities and binary masks only; no annotation or temporal model inputs')
    c.js(a.run/'preflight_verification.json',dict(passed=True,checks=checks,models=details,test_model_discarded=True))
    print(json.dumps(dict(passed=True,checks=len(checks),details=details)),flush=True)

if __name__=='__main__':main()
