"""Framework-independent, original-size RGB prediction/export (Python3.5 compatible)."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import common as c
from comparison import overlay,root_only
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataset_identity import normalize_prediction_io

def predict_files(infer,input_path,output_path,threshold,batch=2):
    input_path=Path(input_path);output_path,_=normalize_prediction_io(input_path,output_path)
    paths=[input_path] if input_path.is_file() else sorted(f for f in input_path.iterdir() if f.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff'])
    assert paths,'No input RGB files';assert len({f.stem for f in paths})==len(paths),'Duplicate input basenames'
    output_path.mkdir(parents=True,exist_ok=True);records=[]
    side=256;han=np.hanning(side).astype(np.float32);blend=np.maximum(han[:,None]*han[None,:],.02)
    for path in paths:
        with Image.open(str(path)) as im:
            assert im.mode=='RGB',('Expected8-bit RGB input; do not silently convert scientific intensities',str(path),im.mode)
            rgb=np.array(im)
        h,w=rgb.shape[:2];ph=max(h,side);pw=max(w,side);padded=np.pad(rgb,((0,ph-h),(0,pw-w),(0,0)),mode='edge')
        xs=list(range(0,pw-side+1,192));ys=list(range(0,ph-side+1,192))
        if xs[-1]!=pw-side:xs.append(pw-side)
        if ys[-1]!=ph-side:ys.append(ph-side)
        coords=[(x,y) for y in ys for x in xs];sm=np.zeros((ph,pw),np.float32);weight=np.zeros_like(sm)
        for i in range(0,len(coords),batch):
            xy=coords[i:i+batch];ims=np.stack([padded[y:y+side,x:x+side] for x,y in xy]).astype(np.float32)/255
            for (x,y),pr in zip(xy,infer(ims)):
                sm[y:y+side,x:x+side]+=pr*blend;weight[y:y+side,x:x+side]+=blend
        assert (weight>0).all();prob=(sm/weight)[:h,:w];mask=prob>=threshold;stem=path.stem
        np.save(str(output_path/(stem+'_probability.npy')),prob)
        Image.fromarray(mask.astype(np.uint8)*255).save(str(output_path/(stem+'_root_mask.png')))
        Image.fromarray(root_only(rgb,mask)).save(str(output_path/(stem+'_root_only_RGB.png')))
        Image.fromarray(overlay(rgb,mask)).save(str(output_path/(stem+'_root_overlay.png')))
        records.append(dict(input_file=str(path),input_SHA256=c.sha(path),size_wh=[w,h],root_pixels=int(mask.sum()),threshold=threshold,
            probability_SHA256=c.sha(output_path/(stem+'_probability.npy'))))
        print('PREDICTED',path.name,w,h,flush=True)
    return records
