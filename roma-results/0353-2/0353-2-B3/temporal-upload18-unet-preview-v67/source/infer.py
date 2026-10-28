"""Read-only U-Net inference from completed V66, never train or alter supports."""
import hashlib,json,sys,time,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path('/workspace/LI')
RUN=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
OUT=ROOT/'results/B3_temporal_upload18_UNet_preview_v67_20261028'
sys.path.insert(0,str(RUN/'source'))
import common as c

def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True);folder=OUT/'unet/full';folder.mkdir(parents=True,exist_ok=True)
    done=json.loads((RUN/'unet/completion.json').read_text());assert done['epochs_completed']==100
    weights=RUN/'unet/best_temporal_head.pt';digest=c.sha(weights);assert digest==done['checkpoint_sha256']
    ck=torch.load(weights,weights_only=True,map_location='cpu');assert ck['epoch']==37 and not ck['config']['GAT']
    model=c.AxisHead().to(memory_format=torch.channels_last);model.load_state_dict(ck['state_dict'])
    prep=json.loads((RUN/'preparation.json').read_text());data=c.Data(RUN,prep['roots'],'unet')
    audit=[];tic=time.monotonic()
    for t in c.PERIODS:
        dest=folder/f'{t}_classes.png'
        if dest.exists():pred=np.array(Image.open(dest))
        elif t=='T1':
            pred=np.where(data.gate[t],4,0).astype(np.uint8);Image.fromarray(pred).save(dest)
        else:
            prob=c.predict_probability(model,data.features[t],torch.device('cpu'))
            assert np.isfinite(prob).all() and np.allclose(prob.sum(0),1,atol=1e-5)
            pred=np.where(data.gate[t],prob.argmax(0)+1,0).astype(np.uint8)
            with (folder/f'{t}_probability.npz').open('wb') as f:np.savez_compressed(f,probability=prob.astype(np.float16))
            Image.fromarray(pred).save(dest)
        assert np.array_equal(pred>0,data.gate[t])
        maskpath=Path(prep['roots'])/'unet/upper'/f'{t}_root_mask.png'
        assert c.sha(maskpath)==next(e['sha256'] for e in prep['immutable_inputs'] if e['path']==str(maskpath))
        audit.append(dict(period=t,root_pixels=int(data.gate[t].sum()),changed_binary_pixels=0,prediction_sha256=c.sha(dest),binary_sha256=c.sha(maskpath)))
        print('INFERRED',t,'elapsed',round(time.monotonic()-tic,2),flush=True)
    assert c.sha(weights)==digest
    c.js(OUT/'inference_verification.json',dict(passed=True,run=str(RUN),checkpoint_SHA256=digest,epochs_completed=100,selected_epoch=37,
        uploaded_reference_SHA256=prep['uploaded_reference_SHA256'],GAT=False,new_training=False,
        changes_to_binary_geometry=False,full_native_dimensions=[2541,1108],periods=audit,
        reference='Uploaded (18) unchanged; T1 baseline display only; same-field internal agreement',
        method='Overlapping 256px patches, Hann blending of softmax; no temporal constraints, class multipliers, dilation or forced matches'))
if __name__=='__main__':main()
