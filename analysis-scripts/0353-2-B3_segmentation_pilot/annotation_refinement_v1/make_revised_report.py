"""Final presentation against revised references, after original-val selection.

Original training/evaluation files are never rewritten. References in this report
are explicitly model-assisted, not independent human ground truth.
"""
import argparse,os,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def link(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():dst.symlink_to(src.resolve(),target_is_directory=src.is_dir())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    data=c.Data(run);view=run/'revised_report';(view/'data').mkdir(parents=True,exist_ok=True)
    rows=[]
    for r in data.rows:
        rr=dict(r)
        link(run/r['RGB_path'],view/r['RGB_path'])
        reference=run/'revision/tile_masks'/r['split']/(r['image_id']+'.png')
        link(reference,view/r['mask_path'])
        g=np.array(Image.open(reference));rr.update(mask_sha256=c.sha(reference),root_pixels=int((g==255).sum()),ignored_pixels=int((g==128).sum()))
        rows.append(rr)
    c.csvout(view/'data/manifest.csv',rows)
    summary=c.read(run/'data/summary.json');summary.update(manifest_SHA256=c.sha(view/'data/manifest.csv'),
        labels='Model-assisted revised references for final presentation; original training-selection audit preserved separately',
        report_reference='Model-assisted revised reference: agreement, not independent manual accuracy',
        reported_revision_reference_is_independent_human_truth=False,
        checkpoint_and_threshold_selection_reference='Original frozen validation masks, not revised validation references',
        original_reference_metrics_path='../tables/original_holdout_metrics.csv')
    c.js(view/'data/summary.json',summary)
    for t in c.PERIODS:
        link(run/'data/upper'/(t+'_RGB.png'),view/'data/upper'/(t+'_RGB.png'))
        link(run/'revision/native'/(t+'_revised_mask.png'),view/'data/upper'/(t+'_mask.png'))
    for name in ['tile_allocation.csv','random_tile_allocation.png','original_human_progress.json']:
        link(run/'data'/name,view/'data'/name)
    composition=[]
    for split in ['train','val','test']:
        rr=[r for r in rows if r['split']==split];roots=sum(r['root_pixels'] for r in rr);valid=sum(65536-r['ignored_pixels'] for r in rr)
        composition.append(dict(split=split,RGB_patch_count=len(rr),physical_tile_positions=len({r['tile_id'] for r in rr}),root_pixels=roots,valid_pixels=valid,root_fraction=roots/valid,rootless_patches=sum(r['root_pixels']==0 for r in rr)))
    c.csvout(view/'data/split_composition.csv',composition)
    reports=[]
    for model in c.MODELS:
        for name in ['best.pt','best_weights_fp16.npz','weight_names.json','config.json','training_run.json','selection.json','history.csv','training_draws.csv','validation_thresholds.csv','upper','probabilities']:
            src=run/model/name
            if src.exists():link(src,view/model/name)
        sel=c.read(run/model/'selection.json');metrics={}
        for split in ['val','test']:
            total=np.zeros(4,np.int64)
            for r in rows:
                if r['split']!=split:continue
                g=np.array(Image.open(view/r['mask_path']));p=np.load(run/model/'probabilities'/split/(r['image_id']+'_probability.npy'))>=sel['threshold']
                total+=c.counts(g,p)
            metrics[split]=c.metrics(total)
            reports.append(dict(model=model,split=split,reference='Revised model-assisted reference; agreement only',**metrics[split]))
        c.js(view/model/'metrics.json',metrics)
    (view/'tables').mkdir(exist_ok=True)
    c.csvout(view/'tables/revised_reference_agreement.csv',reports)
    original=[]
    for model in c.MODELS:
        for split,m in c.read(run/model/'metrics.json').items():
            original.append(dict(model=model,split=split,reference='Original retained human reference; primary internal accuracy audit',**m))
    c.csvout(view/'tables/original_holdout_metrics.csv',original)
    c.js(view/'report_provenance.json',dict(reference='U-Net-assisted revised root masks',
        independent_manual_truth=False,actual_final_model_weights=True,old_models_shown=False,
        original_selection_and_training_outputs_unchanged=True,
        original_holdout_metrics_retained='tables/original_holdout_metrics.csv',
        selection_criterion='Original validation labels only; secondary reference cannot choose weights or threshold',
        root_length='Skeletons derived from revised references; not independent manual root-length measurements',
        S5_colors=dict(TP=[0,158,115],FP=[204,121,167],FN=[230,159,0])))
    print('REVISED_FINAL_REPORT_PREPARED',[(r['model'],r['split'],r['Dice']) for r in reports],flush=True)

if __name__=='__main__':main()
