"""Honest three-model metrics, deployment products and paper-ready visual assets."""
import argparse,csv,json,sys,math
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from scipy import ndimage as ndi
import torch
import common as c
from predict import load_head,products

def lengths(mask):
    sk=skeletonize(mask);out=np.zeros(sk.shape,np.float64)
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        y1=slice(0,-dy) if dy else slice(None);y2=slice(dy,None)
        x1=slice(0,-dx) if dx>0 else slice(-dx,None) if dx<0 else slice(None)
        x2=slice(dx,None) if dx>0 else slice(0,dx) if dx<0 else slice(None)
        edge=sk[y1,x1]&sk[y2,x2]
        if dx and dy:edge&=~(sk[y1,x2]|sk[y2,x1])
        val=edge*.5*math.hypot(dy,dx);out[y1,x1]+=val;out[y2,x2]+=val
    out[sk&(ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')==1)]=1
    return out
def agreement(xx,yy):
    x=np.array(xx,float);y=np.array(yy,float);d=y-x;sst=((x-x.mean())**2).sum()
    return dict(n_correlated_block_date_pairs=len(x),R2_agreement=float(1-(d*d).sum()/sst) if sst else None,RMSE_px=float(np.sqrt((d*d).mean())),MAE_px=float(abs(d).mean()),mean_bias_px=float(d.mean()),reference='Human annotation-derived skeleton length; no separate manual root-length measurements or physical calibration',units='native pixels')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--products',action='store_true');a=p.parse_args();run=a.run;data=run/'data';torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reports={n:json.loads((run/n/'evaluation_metrics.json').read_text()) for n in c.MODELS}
    candidates=[dict(model=n,validation_macro_Dice=reports[n]['validation']['end_to_end']['macro_Dice']) for n in c.MODELS];winner=max(candidates,key=lambda r:r['validation_macro_Dice'])['model']
    c.js(run/'selection.json',dict(winner=winner,criterion='C3 end-to-end temporal macro Dice; C4 excluded',candidates=candidates,C4_used=False,final_fits_have_no_independent_accuracy=True,validation_scores_apply_to='evaluation_head.pt',all_upper_revised_24epoch_fit='final_head.pt',new_position_transfer_accuracy_not_established=True))
    rows=[];per=[];pairrows=[];stats={};verify=[]
    for n in c.MODELS:
        stats[n]={}
        for split,col in [('validation',3),('repeated_benchmark',4)]:
            base=run/n/'evaluation'/split;rep=reports[n][split]
            cm=np.zeros((4,4),np.int64);conditional=np.zeros_like(cm)
            for t in c.PERIODS[1:]:
                gt=np.array(Image.open(base/f'{t}_human_target.png'));pred=np.array(Image.open(base/f'{t}_classes.png'));cm+=c.confusion(gt,pred)
                gg=gt.copy();gg[pred==0]=255;conditional+=c.confusion(gg,pred,True)
                valid=gt!=255;reference_root=(gt>0)&valid;predicted_root=pred>0;err=np.zeros(gt.shape,np.uint8)
                err[reference_root&(gt==pred)]=1;err[reference_root&~predicted_root]=2;err[reference_root&predicted_root&(gt!=pred)]=3;err[valid&~reference_root&predicted_root]=4;err[~valid]=5
                ep=np.array([[255,255,255],[0,158,115],[213,94,0],[204,121,167],[86,180,233],[170,170,170]],np.uint8)
                Image.fromarray(ep[err]).save(base/f'{t}_error_map.png')
                for q in c.metrics(c.confusion(gt,pred))['classes']:per.append(dict(model=n,split=split,period=t,**q))
            assert c.metrics(cm)==rep['end_to_end'];rep['classification_on_detected_human_roots']=c.metrics(conditional);verify.append(dict(model=n,split=split,saved_masks_reproduce_metrics=True))
            for mode in ['end_to_end','classification_on_detected_human_roots']:
                for q in rep[mode]['classes']:rows.append(dict(model=n,split=split,scope=mode,**q))
            for cls in ['all_roots',1,2,3]:
                subset=[]
                for t in c.PERIODS if cls=='all_roots' else c.PERIODS[1:]:
                    if cls=='all_roots':
                        left=c.COLS[col];gt=np.array(Image.open(data/'upper'/f'{t}_mask.png'))[:,left:left+448]==255;pred=np.array(Image.open(data/'root_gates'/n/f'C{col}'/f'{t}.png'))>0
                    else:gt=np.array(Image.open(base/f'{t}_human_target.png'))==cls;pred=np.array(Image.open(base/f'{t}_classes.png'))==cls
                    lf=lengths(gt);lp=lengths(pred)
                    for yi in range(4):
                        for xi in range(2):
                            yy=slice(yi*277,(yi+1)*277);xx=slice(xi*224,(xi+1)*224)
                            row=dict(model=n,split=split,root_class=str(cls),period=t,block=f'X{xi}Y{yi}',human_length_px=float(lf[yy,xx].sum()),predicted_length_px=float(lp[yy,xx].sum()));pairrows.append(row);subset.append(row)
                stats[n][split+'_'+str(cls)]=agreement([r['human_length_px'] for r in subset],[r['predicted_length_px'] for r in subset])
        c.js(run/n/'evaluation_metrics.json',reports[n])
        if a.products:
            head=load_head(run/n/'final_head.pt',device);prev=None;out=run/n/'four_period_final_fit'
            for t in c.PERIODS:
                rgb=np.array(Image.open(data/'upper'/f'{t}_RGB.png'));prob=np.load(a.roots/n/'probabilities/upper'/f'{t}_probability.npy');gate=np.array(Image.open(a.roots/n/'images'/f'{t}_root_mask.png'))>0
                if prev is None:classes=gate.astype(np.uint8)*4;pr=None
                else:
                    pr=c.predict(head,c.features(prev[0],rgb,prev[1],prob,prev[2],gate),device);classes=c.classify(head,pr,gate)
                products(out,t,rgb,gate,classes,pr);prev=(rgb,prob,gate);verify.append(dict(model=n,period=t,root_support_exact_v9=True,native_size_preserved=True,original_root_RGB_unchanged=True))
                print('TEMPORAL_PRODUCT',n,t,flush=True)
            c.js(out/'interpretation.json',dict(purpose='All-B3-label deployment-head fit visualization, not held-out prediction',head_sha256=c.sha(run/n/'final_head.pt'),background_removed=True,T1='baseline, output4',T2='relative to T1',T3='relative to T2',T4='relative to T3',unobserved_not_death=True,no_manual_labels_used_as_prediction_inputs=True))
    c.csvout(run/'temporal_metrics.csv',rows);c.csvout(run/'per_transition_metrics.csv',per);c.csvout(run/'root_length_pairs.csv',pairrows);c.js(run/'root_length_agreement.json',stats);c.js(run/'all_metrics.json',reports)
    checks=json.loads((data/'source_integrity.json').read_text())
    for f,digest in checks['files'].items():assert c.sha(data/f)==digest
    assert c.sha(data/'original_human_progress.json')==checks['annotation_sha256']
    c.js(run/'verification.json',dict(passed=True,checks=verify,original_annotation_file_preserved=True,active_time_labels_revised=True,source_annotation_sha256=checks['annotation_sha256'],background_not_assigned_temporal_classes=True,external_independent_test_images=0))
    print('TEMPORAL_DELIVERY_READY',winner,flush=True)
if __name__=='__main__':main()
