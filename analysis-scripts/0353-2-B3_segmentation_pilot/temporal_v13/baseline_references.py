"""Re-score original and prior-model maps against the revised reference fairly."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
from junctions import junction_region
from revise_annotations import unpack

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();reports={}
    for n in c.MODELS:
        reports[n]={}
        for split,col in [('validation',3),('repeated_benchmark',4)]:
            cm=np.zeros((4,4),np.int64);cc=np.zeros_like(cm);jc=np.zeros_like(cm);periods=[]
            for t in c.PERIODS[1:]:
                gt=np.array(Image.open(a.run/'data/targets'/f'{t}_temporal_target.png'))[:,c.COLS[col]:c.COLS[col]+448];pred=np.array(Image.open(a.previous/n/'evaluation'/split/f'{t}_classes.png'));gate=pred>0;cm+=c.confusion(gt,pred);g=gt.copy();g[~gate]=255;cc+=c.confusion(g,pred,True);g=gt.copy();g[~junction_region(gate)]=255;jc+=c.confusion(g,pred);periods.append(dict(period=t,**c.metrics(c.confusion(gt,pred))))
            reports[n][split]=dict(end_to_end=c.metrics(cm),junction=c.metrics(jc),classification_on_detected_human_roots=c.metrics(cc),by_period=periods)
    c.js(a.run/'baseline_on_revised_labels.json',reports)
    original=json.loads((a.run/'data/original_human_progress.json').read_text());active=json.loads((a.run/'data/active_progress.json').read_text());count=0
    for before,after in zip(original['images'],active['images']):
        for layer in ['mask','color','viability']:assert before['layers'][layer]==after['layers'][layer]
        shape=(before['height'],before['width']);b=unpack(before['layers']['temporal'],shape);r=unpack(after['layers']['temporal'],shape);changed=b!=r;count+=int(changed.sum());assert (b[changed]==1).all() and (r[changed]==3).all();assert not changed[c.HEIGHT:].any()
    assert count==328
    c.js(a.run/'label_revision_verification.json',dict(passed=True,changed_time_pixels=count,original_annotation_sha256=c.sha(a.run/'data/original_human_progress.json'),active_progress_sha256=c.sha(a.run/'data/active_progress.json'),only_new_to_persistent=True,root_background_color_viability_layers_unchanged=True,lower_part_unchanged=True,both_versions_scored_against_same_revised_reference=True,review_origin='Assistant visual revisions authorized by user; not independent human expert adjudication'))
if __name__=='__main__':main()
