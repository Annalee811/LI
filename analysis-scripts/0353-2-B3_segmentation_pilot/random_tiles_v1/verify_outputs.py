"""Integrated artifact checks: native RGB, original labels, mask exports and formats."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();data=c.Data(a.run)
    for r in data.rows:data.load(r)  # Verifies every input RGB and mask SHA, including training.
    checks=dict(input_RGB_and_mask_SHA_checks=2*len(data.rows),local_overlay_background_checks=0,local_root_RGB_checks=0,local_black_root_checks=0,upper_root_RGB_checks=0,upper_black_root_checks=0)
    for r in data.rows:
        if r['split']!='test':continue
        rgb,gt=data.load(r);t=r['period'];tile=r['tile_id'];masks=[gt==255]
        for model in c.MODELS:
            pr=np.load(a.run/model/'probabilities/test'/(r['image_id']+'_probability.npy'));th=c.read(a.run/model/'selection.json')['threshold']
            mask=np.array(Image.open(a.run/'native_predictions'/model/'test'/(r['image_id']+'_prediction.png')))>0
            assert np.array_equal(mask,pr>=th);masks.append(mask)
        folder=a.run/'comparison_RGB/native_local'/tile
        assert np.array_equal(np.array(Image.open(folder/(t+'_RGB.png'))),rgb)
        for name,mask in zip(['Human']+c.MODELS,masks):
            im=np.array(Image.open(folder/(t+'_'+name+'_overlay.png')));assert im.shape==rgb.shape and np.array_equal(im[~mask],rgb[~mask]);checks['local_overlay_background_checks']+=1
            f=a.run/'comparison_RGB/background_removed_local'/tile
            im=np.array(Image.open(f/(t+'_'+name+'_RGB.png')));assert im.shape==rgb.shape and np.array_equal(im[mask],rgb[mask]) and (im[~mask]==255).all();checks['local_root_RGB_checks']+=1
            im=np.array(Image.open(f/(t+'_'+name+'_black_root.png')));assert im.shape==rgb.shape and (im[mask]==0).all() and (im[~mask]==255).all();checks['local_black_root_checks']+=1
    for t in c.PERIODS:
        rgb=np.array(Image.open(a.run/'data/upper'/(t+'_RGB.png')).convert('RGB'));gt=np.array(Image.open(a.run/'data/upper'/(t+'_mask.png')))
        masks=[gt==255]+[np.array(Image.open(a.run/'native_predictions'/m/'upper'/(t+'_root_mask.png')))>0 for m in c.MODELS]
        for name,mask in zip(['Human']+c.MODELS,masks):
            f=a.run/'comparison_RGB/background_removed_upper'/t
            im=np.array(Image.open(f/(name+'_RGB.png')));assert im.shape==rgb.shape and np.array_equal(im[mask],rgb[mask]) and (im[~mask]==255).all();checks['upper_root_RGB_checks']+=1
            im=np.array(Image.open(f/(name+'_black_root.png')));assert im.shape==rgb.shape and (im[mask]==0).all() and (im[~mask]==255).all();checks['upper_black_root_checks']+=1
    caps=c.read(a.run/'figures/captions.json');assert len(caps)==11
    for name in caps:
        for ext in ['png','tif','pdf']:assert (a.run/'figures'/(name+'.'+ext)).stat().st_size>1000
        with Image.open(a.run/'figures'/(name+'.tif')) as im:assert all(abs(v-600)<1 for v in im.info['dpi'])
    assert not list(a.run.rglob('*.docx')) and not list(a.run.rglob('*.doc'))
    checks.update(passed=True,publication_figure_groups=len(caps),all_publication_TIFFs_600dpi=True,Word_files=0,
        original_annotation_SHA256=c.sha(a.run/'data/original_human_progress.json'))
    c.js(a.run/'tables/visual_output_verification.json',checks);print('VISUAL_OUTPUT_VERIFICATION_PASSED',checks,flush=True)

if __name__=='__main__':main()
