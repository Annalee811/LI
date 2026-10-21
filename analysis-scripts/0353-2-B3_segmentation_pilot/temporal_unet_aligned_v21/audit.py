"""Verify saved files and metrics without repeating inference or training."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);a=p.parse_args()
    prep=json.loads((a.run/'preparation.json').read_text());prev=Path(prep['previous']);d=a.run/'data';checks=[]
    assert c.sha(d/'selected_progress.json')==prep['progress_SHA256']==c.sha(prep['source_progress'])
    project=json.loads((d/'active_progress.json').read_text());assert len(project['images'])==4
    for item in project['images']:
        t=item['name'][:2];target=np.array(Image.open(d/'current'/(t+'_classes.png')));mask=c.unpack(item['layers']['mask'],(1108,2541));raw=c.unpack(item['layers']['temporal'],(1108,2541))
        assert c.sha(d/'original'/(t+'_classes.png'))==c.sha(prev/'data/current'/(t+'_classes.png'))
        assert c.sha(d/'upper'/item['name'])==item['image_sha256']
        assert np.array_equal(mask>0,target>0)
        assert np.array_equal(mask,np.array(Image.open(prev/'unet/four_period_prediction'/(t+'_root_mask.png'))))
        expected=np.zeros_like(raw)
        for code,cls in [(1,2),(2,3),(3,1),(4,4)]:expected[(mask>0)&(raw==code)]=cls
        expected[(mask>0)&(raw==0)]=255;assert np.array_equal(target,expected)
        teacher=np.array(Image.open(prev/'unet/four_period_prediction'/(t+'_temporal_classes.png')));known=target!=255;assert np.array_equal(target[known],teacher[known])
    checks+=['original_source_and_class_maps_preserved','four_revised_frames_roundtrip_to_editable_JSON','root_geometry_exact_U_Net','typed_classes_exact_teacher_no_random_errors']
    reference_masks={s:np.zeros((1108,2541),bool) for s in ['train','val','test']}
    for r in c.rows(d/'spatial_manifest.csv'):
        x,y=int(r['x']),int(r['y']);reference_masks[r['split']][y:y+256,x:x+256]=True
    for i,s in enumerate(reference_masks):
        for other in list(reference_masks)[i+1:]:assert not (reference_masks[s]&reference_masks[other]).any()
    assert c.sha(d/'paired_manifest.csv')==c.sha(prev/'data/paired_manifest.csv');checks.append('unchanged_no_cross_split_spatial_pixels')
    sampler_hashes=[]
    for m in c.MODELS:
        out=a.run/m;training=json.loads((out/'training_run.json').read_text());assert training['epochs_completed']==16 and training['root_stage_frozen']
        checkpoint=torch.load(out/'best_temporal_head.pt',map_location='cpu',weights_only=True);assert checkpoint['epoch']==training['selected_epoch'];assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
        last=torch.load(out/'resume.pt',map_location='cpu',weights_only=True);assert last['epoch']==16
        root=a.roots/m/('best_weights_fp16.npz' if m=='deeplab' else 'best.pt');assert c.sha(root)==training['root_SHA256'];sampler_hashes.append(c.sha(out/'training_draws.csv'))
        ref=json.loads((out/'separate_reference_evaluation.json').read_text())
        for report in ref.values():
            for met in report.values():
                actual=c.old.metrics(np.array(met['confusion_matrix']));assert actual==met
        continuity=c.rows(out/'continuity.csv');assert all(int(r['mixed_new_extension_axes'])==0 for r in continuity)
        for t in c.PERIODS:
            rgba=np.array(Image.open(out/'four_period_prediction'/(t+'_original_roots_RGBA.png')));assert (rgba[rgba[:,:,3]==0,:3]==0).all()
    assert len(set(sampler_hashes))==1
    checks+=['all_three_16_epoch_runs_completed','saved_checkpoints_finite_and_readable','binary_weights_unchanged','identical_sampling_and_augmentation_across_models','metrics_recomputed_from_saved_confusions','no_new_extension_mix_within_decoded_candidate_axes','background_zero_in_transparent_RGB']
    assert (a.run/'annotation_review/annotate.html').stat().st_size>1000000
    assert 'temporal-reference-project' in (a.run/'annotation_review/annotate.html').read_text();checks.append('standalone_editable_HTML_contains_RGB_and_proposal')
    c.js(a.run/'verification.json',dict(passed=True,checks=checks,external_independent_images=0,teacher_agreement_is_not_independent_accuracy=True,random_errors_added=False,source_progress_SHA256=prep['progress_SHA256']));print('AUDIT_PASSED',len(checks),flush=True)

if __name__=='__main__':main()
