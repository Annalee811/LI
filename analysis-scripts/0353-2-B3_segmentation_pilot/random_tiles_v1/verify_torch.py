"""Independently reload delivered Torch checkpoints and recheck held-out predictions."""
import argparse
from pathlib import Path
import numpy as np
import torch
import common as c
from models import create

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');data=c.Data(a.run);reports=[]
    for model in ['unet','segformer']:
        folder=a.run/model;ck=torch.load(folder/'best.pt',map_location=device,weights_only=True);net=create(model).to(device);net.load_state_dict(ck['state_dict']);net.eval()
        def infer(ims):
            with torch.inference_mode():return net(torch.from_numpy(ims.transpose(0,3,1,2).copy()).to(device)).sigmoid()[:,0].cpu().numpy()
        selection=c.read(folder/'selection.json');recorded=c.read(folder/'metrics.json');th=selection['threshold']
        for split in ['val','test']:
            maps=data.predict(infer,split,2);largest=0.;different=0
            for key,pr in maps.items():
                old=np.load(folder/'probabilities'/split/(key+'_probability.npy'));largest=max(largest,float(abs(pr-old).max()));different+=int(((pr>=th)!=(old>=th)).sum())
            m,_=data.evaluate(maps,th)
            assert different==0 and all(m[k]==recorded[split][k] for k in ['TP','FP','FN','TN']), (model,split,largest,different)
            reports.append(dict(model=model,split=split,checkpoint_reload_passed=True,largest_probability_difference=largest,changed_mask_pixels=different,counts=m))
        del net
    c.js(a.run/'torch_checkpoint_reload_verification.json',dict(passed=True,device=str(device),reports=reports))
    print('TORCH_CHECKPOINT_RELOAD_PASSED',flush=True)

if __name__=='__main__':main()
