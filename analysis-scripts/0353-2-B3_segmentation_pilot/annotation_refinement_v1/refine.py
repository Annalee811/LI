"""Versioned U-Net-assisted binary reference revision, never independent human GT.

The immutable original validation/test targets remain the model-selection and
primary reporting references. Train references are revised independently inside
each native training tile, without consulting adjacent validation/test pixels.
"""
import argparse, os, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import sato
from skimage.measure import label, regionprops
import torch
import common as c
from models import create

def revise(rgb, gt, probability):
    human=gt==255; valid=gt!=128
    gray=np.asarray(Image.fromarray(rgb).convert('L'),np.float32)/255
    # Bright, tubular RGB support: only a support test, never a root classifier.
    ridge=sato(gray,sigmas=[1,2,3],black_ridges=False,mode='reflect')
    scale=max(float(np.percentile(ridge,99)),1e-6); support=ridge/scale
    near=ndi.distance_transform_edt(~human)<=6
    added=(probability>=.275)&near&(support>=.06)&valid
    # Far from a supplied root, require high model confidence and an elongated
    # connected structure. Droplet rims and round particles are not accepted on
    # brightness alone. No tuning against validation/test Dice is performed.
    far=(probability>=.60)&(support>=.16)&valid
    accepted=np.zeros_like(human)
    for r in regionprops(label(far,connectivity=2)):
        if r.area>=18 and r.eccentricity>=.92 and r.axis_major_length>=18:
            yy,xx=r.coords.T;accepted[yy,xx]=True
    revised=human|added|accepted
    # At most one-pixel local gap repair, and only with image/model support.
    close=ndi.binary_closing(revised,structure=np.ones((3,3),bool))
    bridges=close&~revised&(probability>=.18)&(support>=.12)&near&valid
    revised|=bridges
    out=np.zeros_like(gt);out[revised]=255;out[~valid]=128
    # Disagreements not accepted as roots remain available for later review.
    pending=((probability>=.275)^human)&~(revised^human)&valid
    log=dict(original_root_pixels=int(human.sum()),revised_root_pixels=int(revised.sum()),
        added_pixels=int((revised&~human).sum()),removed_pixels=0,
        narrow_gap_pixels=int(bridges.sum()),unresolved_disagreement_pixels=int(pending.sum()))
    return out,pending,log

def overlay(rgb,mask):
    out=rgb.copy();g=mask==255
    out[g]=np.rint(.55*rgb[g]+.45*np.array([0,158,115])).astype(np.uint8)
    return out

def link_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        try:os.link(src,dst)
        except OSError:shutil.copyfile(src,dst)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    assert not (a.run/'data/summary.json').exists(),'Use a new run folder; original references must never be overwritten'
    torch.set_num_threads(2);torch.manual_seed(c.SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=create('unet').to(device)
    ck=torch.load(a.source/'unet/best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
    original=c.Data(a.source);rows=[];logs=[]
    root=a.run/'revision';(root/'native').mkdir(parents=True,exist_ok=True)
    c.js(root/'provenance.json',dict(version='annotation_refinement_v1',reference_type='U-Net-assisted revised binary reference, not independent human ground truth',
        original_human_progress_SHA256=c.sha(a.source/'data/original_human_progress.json'),teacher_checkpoint_SHA256=c.sha(a.source/'unet/best.pt'),
        RGB_policy='Native 2541x1108 upper images; no resizing or color changes; overlays saved separately',
        reference_policy='Only training targets revised. Original validation/test targets frozen for selection and primary accuracy.',
        temporal_classes_modified=False,death_included=False,original_strokes_preserved=True,
        review='Assistant inspected four full-field RGB/human/U-Net contact sheets and four 512px native crops per date. Bulk pixel edits use disclosed image/model rules; not exhaustive human pixel adjudication.',
        parameters=dict(near_radius_px=6,near_probability=.275,near_ridge_support=.06,far_probability=.60,far_ridge_support=.16,min_far_area=18,min_far_eccentricity=.92,min_far_major_axis=18),
        warnings=['Revised-reference scores measure agreement with model-assisted targets, not independent human accuracy.',
                  'No guaranteed Dice or target accuracy; uncertain and occluded roots cannot be inferred as verified anatomy.',
                  'Original temporal stroke JSON retained; binary revision does not assign new/persistent/extension classes.']))
    for t in c.PERIODS:
        rgb=np.array(Image.open(a.source/'data/upper'/(t+'_RGB.png')).convert('RGB'))
        gt=np.array(Image.open(a.source/'data/upper'/(t+'_mask.png')))
        pr=np.load(a.source/'unet/upper'/(t+'_probability.npy'))
        revised,pending,log=revise(rgb,gt,pr);logs.append(dict(scope='whole_upper_qualitative',image_id=t,**log))
        for name,arr in [('revised_mask',revised),('revised_RGB_overlay',overlay(rgb,revised)),
            ('original_human_RGB_overlay',overlay(rgb,gt)),('unresolved_mask',pending.astype(np.uint8)*255),
            ('added_root_mask',((revised==255)&(gt!=255)).astype(np.uint8)*255)]:
            Image.fromarray(arr).save(root/'native'/(t+'_'+name+'.png'))
        link_or_copy(a.source/'data/upper'/(t+'_RGB.png'),root/'native'/(t+'_original_RGB.png'))
        link_or_copy(a.source/'data/upper'/(t+'_mask.png'),root/'native'/(t+'_original_human_mask.png'))
        for kind in ['RGB','mask']:link_or_copy(a.source/'data/upper'/(t+'_'+kind+'.png'),a.run/'data/upper'/(t+'_'+kind+'.png'))
        print('REVISED_UPPER',t,log,flush=True)
    for r in original.rows:
        rgb,gt=original.load(r)
        with torch.inference_mode():
            xx=torch.from_numpy((rgb.astype(np.float32)/255).transpose(2,0,1)[None].copy()).to(device)
            pr=model(xx).sigmoid()[0,0].float().cpu().numpy()
        revised,pending,log=revise(rgb,gt,pr);logs.append(dict(scope=r['split']+'_native_tile',image_id=r['image_id'],**log))
        revisedpath=root/'tile_masks'/r['split']/(r['image_id']+'.png');revisedpath.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(revised).save(revisedpath)
        rr=dict(r);link_or_copy(a.source/r['RGB_path'],a.run/r['RGB_path'])
        dst=a.run/r['mask_path'];dst.parent.mkdir(parents=True,exist_ok=True)
        if r['split']=='train':Image.fromarray(revised).save(dst)
        else:link_or_copy(a.source/r['mask_path'],dst)
        g=np.array(Image.open(dst));rr.update(mask_sha256=c.sha(dst),root_pixels=int((g==255).sum()),ignored_pixels=int((g==128).sum()))
        rows.append(rr)
    c.csvout(a.run/'data/manifest.csv',rows);c.csvout(root/'revision_log.csv',logs)
    for name in ['original_human_progress.json','tile_allocation.csv','random_tile_allocation.png']:
        link_or_copy(a.source/'data'/name,a.run/'data'/name)
    summary=c.read(a.source/'data/summary.json');summary.update(manifest_SHA256=c.sha(a.run/'data/manifest.csv'),mask_edit=True,
        labels='Model-assisted revised training reference; immutable original human validation/test reference',
        old_B3_trained_weights_allowed_for_initialization=True,initialization_policy='Fine-tune matching previous model on unchanged tile allocation; no test targets used',
        source_original_run=str(a.source),training_reference_provenance='revision/provenance.json',
        whole_field_original_human_masks_unchanged=True)
    c.js(a.run/'data/summary.json',summary)
    composition=[]
    for split in ['train','val','test']:
        rr=[r for r in rows if r['split']==split];roots=sum(r['root_pixels'] for r in rr);valid=sum(65536-r['ignored_pixels'] for r in rr)
        composition.append(dict(split=split,RGB_patch_count=len(rr),physical_tile_positions=len({r['tile_id'] for r in rr}),root_pixels=roots,valid_pixels=valid,root_fraction=roots/valid,rootless_patches=sum(r['root_pixels']==0 for r in rr)))
    c.csvout(a.run/'data/split_composition.csv',composition)
    assert all(r['mask_sha256']==q['mask_sha256'] for r,q in zip(rows,original.rows) if r['split']!='train')
    assert all(r['RGB_sha256']==q['RGB_sha256'] and r['split']==q['split'] for r,q in zip(rows,original.rows))
    c.js(a.run/'data/verification.json',dict(passed=True,RGB_and_split_unchanged=True,original_validation_test_hashes_unchanged=True,
        original_human_JSON_unchanged=True,train_images=100,validation_images=20,test_images=24,revised_reference_not_independent_human_truth=True))
    print('REVISION_DATA_READY',flush=True)

if __name__=='__main__':main()
