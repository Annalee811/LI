"""Check real source/reference hashes, fixed splits, masks and training draws."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    run=a.run;source=Path(c.read(run/'data/summary.json')['source_original_run']);data=c.Data(run);orig=c.Data(source)
    assert len(data.rows)==244 and len(orig.rows)==144
    for r,o in zip(data.rows,orig.rows):
        assert all(r[k]==o[k] for k in ['image_id','tile_id','split','period','x','y','RGB_sha256'])
        if r['split']!='train':assert c.sha(run/r['mask_path'])==c.sha(source/o['mask_path'])
        assert c.sha(run/r['RGB_path'])==c.sha(source/o['RGB_path'])
    assert c.sha(run/'data/original_human_progress.json')==c.sha(source/'data/original_human_progress.json')
    draws=[];checks=[]
    for model in c.MODELS:
        training=c.read(run/model/'training_run.json');selection=c.read(run/model/'selection.json')
        assert training['epochs_completed']==20 and training['optimizer_updates']==640
        assert training['unique_training_images']==200
        ck=run/model/('best_weights_fp16.npz' if model=='deeplab' else 'best.pt')
        assert c.sha(ck)==selection['checkpoint_SHA256']
        d=c.csvread(run/model/'training_draws.csv');assert len(d)==1280
        trainids={r['image_id'] for r in data.rows if r['split']=='train'}
        assert set(r['image_id'] for r in d)==trainids
        draws.append([(int(r['epoch']),int(r['draw']),r['image_id'],r['tile_id']) for r in d])
        for t in c.PERIODS:
            rgb=np.array(Image.open(run/'data/upper'/(t+'_RGB.png')))
            ref=np.array(Image.open(run/'revision/native'/(t+'_revised_mask.png')))
            p=np.load(run/model/'upper'/(t+'_probability.npy'))
            mask=np.array(Image.open(run/model/'upper'/(t+'_root_mask.png')))
            assert rgb.shape==(1108,2541,3) and ref.shape==p.shape==mask.shape==(1108,2541)
            assert np.array_equal(mask>0,p>=selection['threshold'])
            ov=np.array(Image.open(run/'revision/native'/(t+'_revised_RGB_overlay.png')))
            assert np.array_equal(ov[ref!=255],rgb[ref!=255])
            assert c.sha(run/'revision/native'/(t+'_original_RGB.png'))==c.sha(source/'data/upper'/(t+'_RGB.png'))
        checks.append(dict(model=model,epochs_completed=20,optimizer_updates=640,selected_finetune_epoch=selection['selected_epoch'],checkpoint_SHA256=c.sha(ck)))
    assert draws[0]==draws[1]==draws[2]
    assert c.read(run/'torch_checkpoint_reload_verification.json')['passed']
    assert c.read(run/'tensorflow_checkpoint_reload_verification.json')['passed']
    view=run/'revised_report';assert c.read(view/'tables/verification.json')['passed']
    assert c.read(view/'report_provenance.json')['independent_manual_truth'] is False
    expansion=c.read(run/'data/expansion200.json');assert expansion['cross_split_pixel_intersection']==0
    positions={r['tile_id']:r for r in data.rows};rectangle_checks=0
    for r in positions.values():
        assert len([q for q in data.rows if q['tile_id']==r['tile_id']])==4
        for q in positions.values():
            if r['split']==q['split']:continue
            overlap=max(r['x'],q['x'])<min(r['x']+256,q['x']+256) and max(r['y'],q['y'])<min(r['y']+256,q['y']+256)
            assert not overlap,'Cross-split native pixel intersection'
            rectangle_checks+=1
    for r in data.rows[144:]:
        assert r['split']=='train'
        rgb=np.array(Image.open(run/r['RGB_path']));full=np.array(Image.open(run/'data/upper'/(r['period']+'_RGB.png')))
        assert np.array_equal(rgb,full[r['y']:r['y']+256,r['x']:r['x']+256])
    c.js(run/'tables/refinement_verification.json',dict(passed=True,all_244_RGB_checks=True,original_44_validation_test_masks_unchanged=True,
        original_human_JSON_unchanged=True,paired_dates_and_split_unchanged=True,
        actual_three_model_training_draws_equal=True,cross_split_rectangle_checks=rectangle_checks,whole_field_mask_probability_checks=12,
        revised_RGB_background_pixels_unchanged=True,final_revised_report_counts_verified=True,models=checks,reference_revision_is_not_independent_manual_truth=True))
    print('REFINEMENT_VERIFICATION_PASSED',checks,flush=True)

if __name__=='__main__':main()
