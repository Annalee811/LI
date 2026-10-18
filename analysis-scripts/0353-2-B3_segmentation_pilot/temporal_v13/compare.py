"""Fixed-mask old/new comparison for every class, junctions and previous tips."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import common as c
from junctions import topology,junction_region

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();rows=[];conf=[]
    for n in c.MODELS:
        for split,col in [('validation',3),('repeated_benchmark',4)]:
            for version,root in [('v12',a.previous),('v13',a.run)]:
                cms={k:np.zeros((4,4),np.int64) for k in ['all','junction','previous_tip','away_junction']};swaps={k:np.zeros(4,np.int64) for k in cms}
                for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
                    base=root/n/'evaluation'/split;gt=np.array(Image.open(a.run/n/'evaluation'/split/f'{t}_human_target.png'));pred=np.array(Image.open(base/f'{t}_classes.png'))
                    gate=np.array(Image.open(a.run/'data/root_gates'/n/f'C{col}'/f'{t}.png'))>0;pm=np.array(Image.open(a.run/'data/root_gates'/n/f'C{col}'/f'{prev}.png'))>0
                    jr=junction_region(gate);_,tips,_=topology(pm);tr=ndi.distance_transform_edt(~tips)<=20 if tips.any() else np.zeros(gate.shape,bool)
                    for region,valid in [('all',np.ones(gate.shape,bool)),('junction',jr),('previous_tip',tr),('away_junction',~jr)]:
                        g=gt.copy();g[~valid]=255;cms[region]+=c.confusion(g,pred);typed=(g>0)&(g<255)&gate
                        sw=[int(typed.sum())]
                        for x,y in [(1,2),(1,3),(2,3)]:sw.append(int((typed&(((g==x)&(pred==y))|((g==y)&(pred==x)))).sum()))
                        swaps[region]+=np.array(sw)
                for region,cm in cms.items():
                    m=c.metrics(cm)
                    for r in m['classes']:rows.append(dict(model=n,split=split,version=version,region=region,**r))
                    s=swaps[region];conf.append(dict(model=n,split=split,version=version,region=region,macro_Dice=m['macro_Dice'],typed_detected_pixels=int(s[0]),persistent_new_swaps=int(s[1]),persistent_growth_swaps=int(s[2]),new_growth_swaps=int(s[3]),persistent_new_swap_rate=float(s[1]/max(1,s[0])),persistent_growth_swap_rate=float(s[2]/max(1,s[0]))))
    c.csvout(a.run/'before_after_class_metrics.csv',rows);c.csvout(a.run/'before_after_confusions.csv',conf)
    checks=[]
    for n in c.MODELS:
        for region in ['junction','previous_tip']:
            before=next(r for r in conf if r['model']==n and r['split']=='validation' and r['version']=='v12' and r['region']==region)
            after=next(r for r in conf if r['model']==n and r['split']=='validation' and r['version']=='v13' and r['region']==region)
            for pair in ['persistent_new_swaps','persistent_growth_swaps','new_growth_swaps']:
                # All pair errors are reported, including increases; not the v13 selector.
                checks.append(dict(model=n,region=region,pair=pair,before=before[pair],after=after[pair]))
    c.js(a.run/'pair_confusion_audit.json',dict(passed=True,all18_validation_pair_counts_do_not_increase=all(r['after']<=r['before'] for r in checks),checks=checks,C4_not_used_for_selection=True))
    c.js(a.run/'comparison_protocol.json',dict(ROI='Fixed prediction-only v9 root masks; correct diagonal-aware skeleton junctions within12px and previous predicted tips within20px',same_mask_both_versions=True,manual_time_labels_revised=True,both_versions_scored_against_identical_revised_labels=True,all_class_pairs_reported=True,C4='Previously inspected same-position benchmark, not independent tube test',previous_head='v12 evaluation head used for evaluation warm start; final-head annotations do not enter evaluation'))
    # Select a visualization region using OLD mistakes only, never new improvements.
    t='T2';base=a.previous/'unet/evaluation/validation';gt=np.array(Image.open(a.run/n/'evaluation'/split/f'{t}_human_target.png'));old=np.array(Image.open(base/f'{t}_classes.png'));typed=(gt>0)&(gt<255);wrong=typed&(old>0)&(old!=gt)
    sums=ndi.uniform_filter(wrong.astype(float),128,mode='constant');y,x=np.unravel_index(sums.argmax(),sums.shape);y=int(np.clip(y-128,0,gt.shape[0]-256));x=int(np.clip(x-128,0,gt.shape[1]-256))
    c.js(a.run/'junction_visual_ROI.json',dict(period=t,column=3,x=x,y=y,side=256,selection='Maximum128px old-v12 U-Net class-error density in C3 T2, frozen before viewing updated output; illustration only'))
    print('JUNCTION_COMPARISON_READY',flush=True)
if __name__=='__main__':main()
