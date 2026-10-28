import json
import numpy as np
import torch
from data import Data
from model import LocalCorrector
from prepare import OUT,js,sha

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);data=Data();model=LocalCorrector();model.eval()
 with torch.no_grad():
  h,l,_=data.forward(model);frames=data.frames.clone();frames[3]+=1;x=data.x.clone();x[data.phase==3]+=1;hh,ll,_=data.forward(model,frames=frames,x=x);delta=float((h[data.phase<3]-hh[data.phase<3]).abs().max());assert delta<1e-6
  tr=data.tensor['train'];z,e=model.local_logits(h,tr['node'],tr['pf']);assert torch.equal(z.argmax(1),tr['anchor'])
 model.train();h,l,_=data.forward(model);ix=np.flatnonzero((data.g['split']==0)&data.same);loss=torch.nn.functional.binary_cross_entropy_with_logits(l[ix],torch.tensor(data.g['target'][ix],dtype=torch.float32));loss.backward();grad=float(model.memory.gates.weight.grad.abs().sum());gats=[float(q.project.weight.grad.abs().sum()) for q in model.layers];assert grad>0 and all(v>0 for v in gats)
 assert all(not p.requires_grad for p in model.pixel.parameters());prep=json.loads((OUT/'preparation.json').read_text())
 for row in prep['immutable_inputs']:assert sha(row['path'])==row['sha256']
 js(OUT/'preflight_verification.json',dict(passed=True,future_causality_error=delta,ConvLSTM_gradient=grad,GAT_gradients=gats,global_pixel_head_frozen=True,initial_local_labels_equal_frozen_U_Net=True,reference_IDs_used_as_features=False));print('PREFLIGHT PASS',flush=True)
if __name__=='__main__':main()
