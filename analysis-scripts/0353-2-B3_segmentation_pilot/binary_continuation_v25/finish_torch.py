"""One final inference pass for an eligible newly selected Torch checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import torch
import common as c
from models import create

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',choices=['unet','segformer'],required=True);a=p.parse_args();out=a.run/a.model
    if (out/'prediction_complete.json').exists():print('REUSE_COMPLETED_PREDICTIONS',a.model,flush=True);return
    torch.set_num_threads(2);model=create(a.model);model.load_state_dict(torch.load(out/'best.pt',map_location='cpu',weights_only=True)['state_dict']);model.eval();data=c.Data(a.run);threshold=c.read(out/'training_run.json')['threshold']
    def infer(ims):
        with torch.inference_mode():return model(torch.from_numpy(ims.transpose(0,3,1,2).copy())).sigmoid()[:,0].float().numpy()
    reports={}
    for split in ['val','test']:
        maps=data.predict(infer,split,2);c.save_probabilities(out/'probabilities'/split,maps);reports[split],rr=data.evaluate(maps,threshold);c.csvout(out/(split+'_per_tile_metrics.csv'),rr)
    c.js(out/'metrics.json',reports);c.full_predict(infer,a.run,out/'upper',threshold,2);c.js(out/'prediction_complete.json',dict(state='complete',checkpoint_SHA256=c.sha(out/'best.pt'),test_used_for_training=False));print('PREDICTIONS_COMPLETE',a.model,flush=True)

if __name__=='__main__':main()
