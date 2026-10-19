"""Native RGB inference for additional positions with a frozen trained Torch model."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from models import create
from comparison import overlay,root_only
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataset_identity import normalize_prediction_io,policy

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['unet','segformer'],required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--batch',type=int,default=2);a=p.parse_args()
    a.output,position_id=normalize_prediction_io(a.input,a.output)
    torch.set_num_threads(a.threads);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sel=c.read(a.model_dir/'selection.json');th=sel['threshold'];checkpoint=a.model_dir/'best.pt'
    assert c.sha(checkpoint)==sel['checkpoint_SHA256'];net=create(a.model).to(device)
    net.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)['state_dict']);net.eval()
    paths=[a.input] if a.input.is_file() else sorted(f for f in a.input.iterdir() if f.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff'])
    assert paths,'No input RGB files';a.output.mkdir(parents=True,exist_ok=True);records=[]
    side=256;han=np.hanning(side).astype(np.float32);blend=np.maximum(han[:,None]*han[None,:],.02)
    for path in paths:
        with Image.open(path) as im:
            assert im.mode=='RGB',('Expected 8-bit RGB input; do not silently convert scientific intensity data',path,im.mode)
            rgb=np.array(im)
        h,w=rgb.shape[:2];ph=max(h,side);pw=max(w,side)
        padded=np.pad(rgb,((0,ph-h),(0,pw-w),(0,0)),mode='edge')
        xs=list(range(0,pw-side+1,192));ys=list(range(0,ph-side+1,192))
        if xs[-1]!=pw-side:xs.append(pw-side)
        if ys[-1]!=ph-side:ys.append(ph-side)
        coords=[(x,y) for y in ys for x in xs];sm=np.zeros((ph,pw),np.float32);weight=np.zeros_like(sm)
        for i in range(0,len(coords),a.batch):
            xy=coords[i:i+a.batch];ims=np.stack([padded[y:y+side,x:x+side] for x,y in xy]).astype(np.float32)/255
            with torch.inference_mode():pr=net(torch.from_numpy(ims.transpose(0,3,1,2).copy()).to(device)).sigmoid()[:,0].cpu().numpy()
            for (x,y),q in zip(xy,pr):sm[y:y+side,x:x+side]+=q*blend;weight[y:y+side,x:x+side]+=blend
        assert (weight>0).all();prob=(sm/weight)[:h,:w];mask=prob>=th;stem=path.stem
        np.save(a.output/(stem+'_probability.npy'),prob)
        Image.fromarray(mask.astype(np.uint8)*255).save(a.output/(stem+'_root_mask.png'))
        Image.fromarray(root_only(rgb,mask)).save(a.output/(stem+'_root_only_RGB.png'))
        Image.fromarray(overlay(rgb,mask)).save(a.output/(stem+'_root_overlay.png'))
        records.append(dict(input_file=str(path),input_SHA256=c.sha(path),size_wh=[w,h],root_pixels=int(mask.sum()),
            threshold=th,probability_SHA256=c.sha(a.output/(stem+'_probability.npy'))))
        print('PREDICTED',path.name,w,h,flush=True)
    c.js(a.output/'prediction_provenance.json',dict(model=a.model,position_id=position_id,dataset_identity_policy=policy()['version'],checkpoint_SHA256=c.sha(checkpoint),device=str(device),
        threshold_selected_on='Original training-run validation only; not adapted on these images',RGB_resized=False,
        RGB_recolored=False,background_only_changed_in_overlay_and_root_only_visuals=True,
        evaluation_on_new_images='Not measured without corresponding human labels',files=records))

if __name__=='__main__':main()
