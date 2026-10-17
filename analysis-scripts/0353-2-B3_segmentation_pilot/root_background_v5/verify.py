#!/usr/bin/env python3
"""Verify source labels, grouped native tiles, and actual RGB-only inference."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_uploaded_upper import SmallUNet

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--rgb-only-checkpoint',type=Path,required=True);a=p.parse_args()
    data=c.Data(a.run);progress=json.loads((a.run/'original_human_progress.json').read_text());n=0
    for item in progress['images']:
        t=item['name'][:2]
        for key,rle in item['layers'].items():
            original=np.repeat(np.asarray(rle[::2],np.uint8),rle[1::2]).reshape(1867,2541)[:1108].copy()
            if key=='mask':original[original==64]=0
            assert np.array_equal(original,np.array(Image.open(a.run/'dataset/upper'/(t+'_'+key+'.png'))));n+=1
    with (a.run/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        x,y=int(r['x']),int(r['y']);t=r['period']
        for key in ['RGB','mask','color','temporal','viability']:
            path=a.run/r[key+'_path'];assert c.sha(path)==r[key+'_sha256']
            source=np.array(Image.open(a.run/'dataset/upper'/(t+'_'+key+'.png')))
            assert np.array_equal(np.array(Image.open(path)),source[y:y+256,x:x+256]);n+=1
        peers=[q for q in rows if q['window_id']==r['window_id']]
        assert {q['period'] for q in peers}==set(c.PERIODS) and len({q['split'] for q in peers})==1
    torch.set_num_threads(1);ck=torch.load(a.rgb_only_checkpoint,map_location='cpu',weights_only=True)
    model=SmallUNet(ck['width']);model.load_state_dict(ck['state_dict']);model.eval()
    inputs=[]
    def infer(ims):
        inputs.append(hashlib.sha256(ims.tobytes()).hexdigest())
        with torch.inference_mode():return model(torch.from_numpy(ims.transpose(0,3,1,2).copy())).sigmoid()[:,0].numpy()
    before=data.column(infer,3);beforeinputs=inputs.copy();inputs.clear()
    # Destroy all loaded manual masks and root-centred coordinate pools; inference must be unchanged.
    for t in c.PERIODS:data.gt[t][:]=255
    data.rootcoords={}
    after=data.column(infer,3)
    assert beforeinputs==inputs and all(np.array_equal(before[t],after[t]) for t in c.PERIODS)
    c.js(a.run/'input_verification.json',dict(passed=True,layer_and_tile_checks=n,
        source_progress_sha256=c.sha(a.run/'original_human_progress.json'),
        manual_labels_mutated_without_changing_actual_model_predictions=True,
        exact_inference_tensor_hashes_unchanged=True,periods_checked=c.PERIODS,
        model_used_for_control='Previously trained3-channel RGB-only U-Net',control_checkpoint_sha256=c.sha(a.rgb_only_checkpoint)))
    print('ROOT_INPUT_VERIFIED',n,flush=True)
if __name__=='__main__':main()
