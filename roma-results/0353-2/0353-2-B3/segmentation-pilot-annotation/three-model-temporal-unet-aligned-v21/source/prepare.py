"""Full-field teacher proposal; immutable original annotations retained for audit."""
import argparse, copy, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    p=argparse.ArgumentParser()
    for key in ['run','roots','previous','progress']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    if (a.run/'preparation.json').exists():
        assert json.loads((a.run/'preparation.json').read_text())['progress_SHA256']==c.sha(a.progress)
        print('REUSE_PREPARATION',flush=True);return
    d=a.run/'data'
    for f in ['upper','current','original','revision']:(d/f).mkdir(parents=True,exist_ok=True)
    source=json.loads(a.progress.read_text());items={i['name']:i for i in source['images']}
    active=copy.deepcopy(source);active['images']=[];full=copy.deepcopy(source)
    shutil.copy2(a.progress,d/'selected_progress.json')
    for name in ['paired_manifest.csv','spatial_manifest.csv']:shutil.copy2(a.previous/'data'/name,d/name)
    domains={s:np.zeros((1108,2541),bool) for s in ['train','val','test']}
    for r in c.rows(d/'spatial_manifest.csv'):
        x,y=int(r['x']),int(r['y']);domains[r['split']][y:y+256,x:x+256]=True
    assert not (domains['train']&(domains['val']|domains['test'])).any()
    Image.fromarray(domains['train'].astype('uint8')*255).save(d/'revision/training_domain_mask.png')
    teacher=a.previous/'unet/four_period_prediction';summaries=[];log=[]
    for t in c.PERIODS:
        item=items[t+'_RGB.png'];shape=(item['height'],item['width']);assert shape==(1108,2541)
        rgb=a.previous/'data/upper'/item['name'];assert c.sha(rgb)==item['image_sha256'];shutil.copy2(rgb,d/'upper'/item['name'])
        original=np.array(Image.open(a.previous/'data/current'/(t+'_classes.png')))
        Image.fromarray(original).save(d/'original'/(t+'_classes.png'))
        oldmask=c.unpack(item['layers']['mask'],shape);oldraw=c.unpack(item['layers']['temporal'],shape)
        target=np.array(Image.open(teacher/(t+'_temporal_classes.png')));mask=np.array(Image.open(teacher/(t+'_root_mask.png')))
        assert set(np.unique(target)).issubset({0,1,2,3,4})
        assert np.array_equal(mask>0,target>0)
        review=np.zeros(shape,bool)
        if t=='T1':target=np.where(mask>0,4,0).astype('uint8')
        else:
            with np.load(teacher/(t+'_raw_probability.npz')) as z:prob=z['probability'].astype('float32')
            sorted_prob=np.sort(prob,axis=0);margin=sorted_prob[-1]-sorted_prob[-2]
            assigned=np.take_along_axis(prob,np.clip(target.astype(int)-1,0,2)[None],0)[0]
            junction=np.array(Image.open(teacher/(t+'_crossing_review_mask.png')))>0
            # Do not invent label errors. Only explicitly ambiguous, weakly supported
            # crossing pixels are untyped, keeping a single categorical owner elsewhere.
            review=(mask>0)&junction&(assigned<.5)&(margin<.1)
            target[review]=255
        raw=np.zeros(shape,np.uint8)
        for cls,code in [(1,3),(2,1),(3,2),(4,4)]:raw[target==cls]=code
        changed=target!=original;geometry=(oldmask==255)!=(mask>0)
        for layer,array in [('mask',mask),('temporal',raw)]:Image.fromarray(array).save(d/'upper'/(t+'_'+layer+'.png'))
        Image.fromarray(target).save(d/'current'/(t+'_classes.png'))
        for name,array in [('changed_pixels',changed),('geometry_changed_pixels',geometry),('review_pixels',review)]:
            Image.fromarray(array.astype('uint8')*255).save(d/'revision'/(t+'_'+name+'.png'))
        Image.fromarray(raw).save(d/'revision'/(t+'_temporal_revised.png'))
        updated=copy.deepcopy(item);updated['layers']['mask']=c.pack(mask);updated['layers']['temporal']=c.pack(raw);updated['reviewed']=False
        active['images'].append(updated)
        for idx,old in enumerate(full['images']):
            if old['name']==item['name']:full['images'][idx]=copy.deepcopy(updated)
        counts=dict(period=t,changed_pixels=int(changed.sum()),geometry_changed_pixels=int(geometry.sum()),review_pixels=int(review.sum()),root_pixels=int((mask>0).sum()),teacher_class_agreement_on_typed_roots=float(np.mean(target[(mask>0)&~review]==np.array(Image.open(teacher/(t+'_temporal_classes.png')))[(mask>0)&~review])),**{s+'_changed_pixels':int((changed&domain).sum()) for s,domain in domains.items()})
        summaries.append(counts)
        log.append(dict(**counts,original_classes_SHA256=c.sha(d/'original'/(t+'_classes.png')),revised_classes_SHA256=c.sha(d/'current'/(t+'_classes.png')),teacher_classes_SHA256=c.sha(teacher/(t+'_temporal_classes.png')),reason='Full U-Net root geometry and structured temporal proposal; weak ambiguous junctions untyped',human_reviewed=False))
    policy=dict(reference_type='Full-field U-Net-derived temporal annotation proposal, pending user review',source_progress_SHA256=c.sha(a.progress),teacher_temporal_head_SHA256=c.sha(a.previous/'unet/best_temporal_head.pt'),teacher_binary_checkpoint_SHA256=c.sha(a.roots/'unet/best.pt'),scope='All four upper RGB frames; no lower frames',original_labels_preserved=True,root_geometry='U-Net frozen binary root masks',semantics='T1 baseline; subsequent persistent shaft, terminal extension, newly observed root/side root relative to adjacent previous period',ambiguity_rule='Only teacher-flagged crossing root pixels with assigned score <0.5 AND top-two margin <0.1 are temporally untyped; no background removal there',random_label_perturbation=False,human_reviewed=False,validation_reference='Teacher-derived proposal; consistency only, not independent accuracy',original_audit_reference='Byte-preserved user-selected (6) labels; historical same-field model exposure',historical_same_field_exposure=True,external_independent_test_images=0)
    active['annotation_revision']=policy;full['annotation_revision']=policy
    c.js(d/'active_progress.json',active);c.js(d/'full_progress_compatibility.json',full);c.js(d/'revision_provenance.json',policy);c.js(d/'revision_summary.json',summaries);c.csvout(d/'revision_summary.csv',summaries);c.js(d/'revision_log.json',log)
    # Geometry/features were computed from frozen roots and RGB, never target
    # class pixels. Reuse the exact V19 train-only routing for all three branches.
    for folder in ['routing','axis_cache','feature_cache']:
        dest=a.run/folder
        if not dest.exists():dest.symlink_to((a.previous/folder).resolve(),target_is_directory=True)
    pairs=c.rows(d/'paired_manifest.csv')
    config=dict(version='unet_aligned_v21',roots=str(a.roots.resolve()),previous=str(a.previous.resolve()),source_progress=str(a.progress.resolve()),progress_SHA256=c.sha(a.progress),training_reference_SHA256=c.sha(d/'active_progress.json'),revision_policy=policy,train_val_test_pairs={s:sum(r['split']==s for r in pairs) for s in domains},total_changed_pixels=sum(r['changed_pixels'] for r in summaries),total_review_pixels=sum(r['review_pixels'] for r in summaries),frozen_original_audit=True,root_networks_frozen=True,feature_and_geometry_cache='Exact V19 RGB/root-only caches and V19 train-only routers reused',death_included=False,independent_external_test_images=0)
    c.js(a.run/'preparation.json',config);print('FULL_REFERENCE_READY',summaries,flush=True)

if __name__=='__main__':main()
