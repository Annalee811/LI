"""Label-free pair prediction after each model's root/background segmentation."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from model import TemporalHead
for identity_parent in [Path(__file__).resolve().parents[2],Path(__file__).resolve().parents[1]]:
    sys.path.insert(0,str(identity_parent))
from dataset_identity import normalize_prediction_io,policy

def load_head(path,device):
    ck=torch.load(path,map_location=device,weights_only=True);m=TemporalHead().to(device,memory_format=torch.channels_last);m.load_state_dict(ck['state_dict'])
    if 'class_multipliers' in ck:m.class_multipliers=ck['class_multipliers']
    if 'class_adjustment' in ck:m.class_adjustment=ck['class_adjustment']
    m.chain_penalty=ck.get('chain_penalty',0.);return m.eval()

def products(out,period,rgb,gate,classes,probability=None):
    out.mkdir(parents=True,exist_ok=True);assert rgb.shape[:2]==classes.shape==gate.shape
    assert np.array_equal(classes>0,gate),'Temporal stage must preserve exactly the frozen root support'
    Image.fromarray(classes).save(out/f'{period}_temporal_classes.png')
    Image.fromarray(gate.astype(np.uint8)*255).save(out/f'{period}_root_mask.png')
    Image.fromarray(c.PALETTE[classes]).save(out/f'{period}_temporal_on_white.png')
    rgba=np.concatenate([rgb,gate[:,:,None].astype(np.uint8)*255],2);rgba[~gate,:3]=0
    Image.fromarray(rgba).save(out/f'{period}_original_roots_RGBA.png')
    color_rgba=np.concatenate([c.PALETTE[classes],gate[:,:,None].astype(np.uint8)*255],2);color_rgba[~gate,:3]=0
    Image.fromarray(color_rgba).save(out/f'{period}_temporal_RGBA.png')
    overlay=rgb.copy();overlay[gate]=np.rint(.4*rgb[gate]+.6*c.PALETTE[classes[gate]]).astype(np.uint8)
    Image.fromarray(overlay).save(out/f'{period}_temporal_overlay.png')
    Image.fromarray(rgb).save(out/f'{period}_RGB.png')
    assert np.array_equal(rgba[gate,:3],rgb[gate])
    if probability is not None:
        masked=probability*gate[None];np.savez_compressed(out/f'{period}_class_probability.npz',probability=masked.astype(np.float16))
        conf=np.take_along_axis(masked,np.clip(classes.astype(int)-1,0,2)[None],axis=0)[0];Image.fromarray(np.rint(conf*65535).astype(np.uint16)).save(out/f'{period}_class_confidence_uint16.png')

def main():
    p=argparse.ArgumentParser();p.add_argument('--head',type=Path,required=True);p.add_argument('--rgb-dir',type=Path,required=True);p.add_argument('--root-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output,position_id=normalize_prediction_io(a.rgb_dir,a.output)
    torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=load_head(a.head,device)
    files={t:a.rgb_dir/f'{t}_RGB.png' for t in c.PERIODS};records=[];previous=None
    for t in c.PERIODS:
        if not files[t].exists():continue
        rgb=np.array(Image.open(files[t]).convert('RGB'));prob=np.load(a.root_dir/f'{t}_probability.npy');gate=np.array(Image.open(a.root_dir/f'{t}_root_mask.png'))>0
        assert rgb.shape[:2]==prob.shape==gate.shape
        if previous is None:
            assert t=='T1','Baseline T1 is required; no invented transition across a missing date'
            classes=gate.astype(np.uint8)*4;pr=None
        else:
            pt,pi,pp,pm=previous;assert c.PERIODS.index(t)-c.PERIODS.index(pt)==1,'Missing intermediate period: provide a declared nonadjacent comparison separately'
            x=c.features(pi,rgb,pp,prob,pm,gate);pr=c.predict(model,x,device);classes=c.classify(model,pr,gate)
        products(a.output,t,rgb,gate,classes,pr);records.append(dict(period=t,previous=previous[0] if previous else None,RGB_sha256=c.sha(files[t]),root_mask_sha256=c.sha(a.root_dir/f'{t}_root_mask.png'),size=[rgb.shape[1],rgb.shape[0]],human_labels_read=False,class_pixels={str(k):int((classes==k).sum()) for k in [1,2,3,4]}));previous=(t,rgb,prob,gate)
    c.js(a.output/'inference.json',dict(position_id=position_id,dataset_identity_policy=policy()['version'],head_sha256=c.sha(a.head),manual_masks_used=False,background_removed_before_temporal_head=True,unobserved_is_not_proven_death=True,records=records))
if __name__=='__main__':main()
