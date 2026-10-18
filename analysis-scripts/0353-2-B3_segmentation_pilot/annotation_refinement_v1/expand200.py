"""Add one native 256px translated training view per original train tile/date.

The extra view is shifted by12pixels in x/y, inward at image boundaries.
All original val/test pixels are excluded by explicit rectangle intersection.
This yields200 training views, not200 independent original acquisitions.
"""
import argparse,itertools
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from models import create
from refine import revise

def intersects(r,s):
    return max(int(r['x']),int(s['x']))<min(int(r['x'])+256,int(s['x'])+256) and max(int(r['y']),int(s['y']))<min(int(r['y'])+256,int(s['y'])+256)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    data=c.Data(run);source=Path(c.read(run/'data/summary.json')['source_original_run'])
    assert len(data.rows)==144 and sum(r['split']=='train' for r in data.rows)==100
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=create('unet').to(device);net.load_state_dict(torch.load(source/'unet/best.pt',map_location=device,weights_only=True)['state_dict']);net.eval()
    tiles=c.csvread(run/'data/tile_allocation.csv');held=[r for r in tiles if r['split']!='train'];newtiles=[];newrows=[];logs=c.csvread(run/'revision/revision_log.csv')
    arrays={t:(np.array(Image.open(run/'data/upper'/(t+'_RGB.png'))),np.array(Image.open(run/'data/upper'/(t+'_mask.png')))) for t in c.PERIODS}
    for tile in tiles:
        if tile['split']!='train':continue
        x,y=int(tile['x']),int(tile['y']);sx=12 if x+256+12<=2541 else -12;sy=12 if y+256+12<=1108 else -12
        nt=dict(tile_id=tile['tile_id']+'_S12',x=x+sx,y=y+sy,width=256,height=256,split='train')
        assert all(not intersects(nt,h) for h in held),'Shifted training tile intersects held-out pixels'
        newtiles.append(nt)
        for t in c.PERIODS:
            rgb,gt=arrays[t];xx,yy=nt['x'],nt['y'];rgb=rgb[yy:yy+256,xx:xx+256];gt=gt[yy:yy+256,xx:xx+256]
            with torch.inference_mode():
                p=net(torch.from_numpy((rgb.astype(np.float32)/255).transpose(2,0,1)[None].copy()).to(device)).sigmoid()[0,0].cpu().numpy()
            revised,pending,log=revise(rgb,gt,p);key=nt['tile_id']+'_'+t
            paths={}
            for kind,arr in [('RGB',rgb),('mask',revised)]:
                dest=run/'data/patches/train'/kind/(key+'.png');Image.fromarray(arr).save(dest)
                paths[kind+'_path']=str(dest.relative_to(run));paths[kind+'_sha256']=c.sha(dest)
            Image.fromarray(revised).save(run/'revision/tile_masks/train'/(key+'.png'))
            newrows.append(dict(image_id=key,period=t,**nt,**paths,root_pixels=int((revised==255).sum()),ignored_pixels=int((revised==128).sum())))
            logs.append(dict(scope='train_shifted_native_tile',image_id=key,**log))
    assert len(newrows)==100 and len(newtiles)==25
    rows=data.rows+newrows;c.csvout(run/'data/manifest.csv',rows)
    # csvread coordinates are strings, explicit field order remains unchanged.
    c.csvout(run/'data/tile_allocation.csv',tiles+newtiles);c.csvout(run/'revision/revision_log.csv',logs)
    s=c.read(run/'data/summary.json');s.update(manifest_SHA256=c.sha(run/'data/manifest.csv'),
        RGB_images=dict(train=200,val=20,test=24),spatial_tiles=dict(train=50,val=5,test=6),
        unique_base_random_tile_positions=dict(train=25,val=5,test=6),
        augmentation_training_views='One additional256px native view translated12px in x/y per training base position, inward at boundaries; views overlap within train only',
        total_native_views=244,independent_original_RGB_images=4,
        added_train_views_are_independent_acquisitions=False,all_added_views_exclude_original_val_test_pixels=True)
    c.js(run/'data/summary.json',s)
    comp=[]
    for split in ['train','val','test']:
        rr=[r for r in rows if r['split']==split];roots=sum(int(r['root_pixels']) for r in rr);valid=sum(65536-int(r['ignored_pixels']) for r in rr)
        comp.append(dict(split=split,RGB_patch_count=len(rr),physical_tile_positions=len({r['tile_id'] for r in rr}),root_pixels=roots,valid_pixels=valid,root_fraction=roots/valid,rootless_patches=sum(int(r['root_pixels'])==0 for r in rr)))
    c.csvout(run/'data/split_composition.csv',comp)
    verification=c.read(run/'data/verification.json');verification.update(train_images=200,total_tile_date_views=244,
        added_training_view_vs_heldout_rectangle_checks=25*11,added_views_overlap_only_within_training=True)
    c.js(run/'data/verification.json',verification)
    c.js(run/'data/expansion200.json',dict(train_views=200,val_views=20,test_views=24,new_view_positions=25,
        shift_px=12,heldout_reference_unchanged=True,overlap_between_training_views=True,
        cross_split_pixel_intersection=0,claim='200 training views from25base positions; not200independent acquisitions'))
    print('TRAINING_EXPANDED_TO_200; ORIGINAL_VALIDATION_20_TEST_24_PRESERVED',flush=True)

if __name__=='__main__':main()
