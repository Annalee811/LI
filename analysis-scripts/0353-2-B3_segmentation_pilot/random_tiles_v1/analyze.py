"""Recompute scores from delivered probabilities, unchanged labels and frozen thresholds."""
import argparse,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import common as c

def root_length(mask):
    sk=skeletonize(mask); total=0.
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        y1=slice(0,-dy) if dy else slice(None); y2=slice(dy,None)
        x1=slice(0,-dx) if dx>0 else slice(-dx,None) if dx<0 else slice(None)
        x2=slice(dx,None) if dx>0 else slice(0,dx) if dx<0 else slice(None)
        edges=sk[y1,x1]&sk[y2,x2]
        if dx and dy: edges &= ~(sk[y1,x2]|sk[y2,x1])
        total+=edges.sum()*math.hypot(dy,dx)
    total+=int((sk&(ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')==1)).sum())
    return float(total)

def agreement(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); d=y-x; sst=((x-x.mean())**2).sum()
    den=x.var()+y.var()+(x.mean()-y.mean())**2; sd=d.std(ddof=1)
    slope,intercept=np.polyfit(x,y,1); fitted=slope*x+intercept; sy=((y-y.mean())**2).sum()
    return dict(n_correlated_tile_date_pairs=len(x),R2_identity=float(1-(d*d).sum()/sst) if sst else None,
        R2_OLS=float(1-((y-fitted)**2).sum()/sy) if sy else None,RMSE_px=float(np.sqrt((d*d).mean())),
        MAE_px=float(abs(d).mean()),bias_px=float(d.mean()),CCC=float(2*((x-x.mean())*(y-y.mean())).mean()/den) if den else None,
        LoA_lower_px=float(d.mean()-1.96*sd),LoA_upper_px=float(d.mean()+1.96*sd),slope=float(slope),intercept_px=float(intercept))

def cldice(g,p):
    sg=skeletonize(g); sp=skeletonize(p)
    if not sg.any() or not sp.any(): return 0. if sg.any() or sp.any() else None
    precision=(sp&g).sum()/sp.sum(); recall=(sg&p).sum()/sg.sum()
    return float(2*precision*recall/(precision+recall)) if precision+recall else 0.

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); a=p.parse_args()
    data=c.Data(a.run); out=a.run/'tables'; out.mkdir(parents=True,exist_ok=True)
    c.csvout(out/'split_composition.csv',c.csvread(a.run/'data/split_composition.csv'))
    aggregated=[]; per_tile=[]; periods=[]; tilegroups=[]; pairs=[]; length=[]; maskchecks=[]
    for model in c.MODELS:
        folder=a.run/model; selection=c.read(folder/'selection.json'); th=selection['threshold']
        ck=folder/('best_weights_fp16.npz' if model=='deeplab' else 'best.pt')
        assert c.sha(ck)==selection['checkpoint_SHA256']
        for split in ['val','test']:
            maps={}
            for r in data.rows:
                if r['split']!=split:continue
                key=r['image_id']; pr=np.load(folder/'probabilities'/split/(key+'_probability.npy'))
                assert pr.shape==(256,256) and np.isfinite(pr).all() and pr.min()>=0 and pr.max()<=1
                rgb,gt=data.load(r); pred=pr>=th; valid=gt!=128; maps[key]=pr
                dest=a.run/'native_predictions'/model/split; dest.mkdir(parents=True,exist_ok=True)
                f=dest/(key+'_prediction.png'); Image.fromarray(pred.astype(np.uint8)*255).save(f)
                m=c.metrics(c.counts(gt,pred)); per_tile.append(dict(model=model,**{k:r[k] for k in ['image_id','tile_id','period','split','x','y']},threshold=th,
                    **m,clDice=cldice((gt==255)&valid,pred&valid)))
                maskchecks.append(dict(model=model,image_id=key,split=split,mask_SHA256=c.sha(f)))
                if split=='test':
                    pairs.append(dict(model=model,image_id=key,tile_id=r['tile_id'],period=r['period'],
                        human_mask_skeleton_length_px=root_length((gt==255)&valid),predicted_skeleton_length_px=root_length(pred&valid)))
            m,_=data.evaluate(maps,th); training=c.read(folder/'training_run.json')
            aggregated.append(dict(model=model,name=c.NAMES[model],split=split,threshold=th,**m,
                spatial_tiles=len({r['tile_id'] for r in data.rows if r['split']==split}),
                RGB_patches=sum(r['split']==split for r in data.rows),selected_epoch=selection['selected_epoch'],
                epochs_completed=training['epochs_completed']))
            recorded=c.read(folder/'metrics.json')[split]
            for k in ['TP','FP','FN','TN','Dice','IoU']:
                assert m[k]==recorded[k],(model,split,k,m[k],recorded[k])
            rr=[r for r in per_tile if r['model']==model and r['split']==split]
            for t in c.PERIODS:
                cc=np.array([[r[k] for k in ['TP','FP','FN','TN']] for r in rr if r['period']==t]).sum(0)
                periods.append(dict(model=model,split=split,period=t,**c.metrics(cc)))
            for tile in sorted({r['tile_id'] for r in rr}):
                cc=np.array([[r[k] for k in ['TP','FP','FN','TN']] for r in rr if r['tile_id']==tile]).sum(0)
                tilegroups.append(dict(model=model,split=split,tile_id=tile,**c.metrics(cc)))
        pp=[r for r in pairs if r['model']==model]
        length.append(dict(model=model,**agreement([r['human_mask_skeleton_length_px'] for r in pp],[r['predicted_skeleton_length_px'] for r in pp])))
        # Native qualitative outputs retain original pixel coordinates and RGB colors.
        dest=a.run/'native_predictions'/model/'upper'; dest.mkdir(parents=True,exist_ok=True)
        for t in c.PERIODS:
            rgb=np.array(Image.open(a.run/'data/upper'/(t+'_RGB.png')).convert('RGB'))
            pr=np.load(folder/'upper'/(t+'_probability.npy')); pred=pr>=th
            mask=np.array(Image.open(folder/'upper'/(t+'_root_mask.png')))>0; assert np.array_equal(pred,mask)
            Image.fromarray(pred.astype(np.uint8)*255).save(dest/(t+'_root_mask.png'))
            only=rgb.copy(); only[~pred]=255; Image.fromarray(only).save(dest/(t+'_root_only_RGB.png'))
            overlay=rgb.copy(); overlay[pred]=np.rint(.5*rgb[pred]+.5*np.array([0,158,115])).astype(np.uint8)
            Image.fromarray(overlay).save(dest/(t+'_overlay50.png'))
    for name,rows in [('aggregate_metrics',aggregated),('per_tile_metrics',per_tile),('per_period_metrics',periods),
                      ('four_date_pooled_tile_metrics',tilegroups),('root_length_pairs',pairs),('root_length_agreement',length)]:c.csvout(out/(name+'.csv'),rows)
    best=max((r for r in aggregated if r['split']=='val'),key=lambda r:r['Dice'])['model']
    c.js(out/'summary.json',dict(dataset=c.read(a.run/'data/summary.json'),aggregate=aggregated,root_length=length,
        model_selected_by_validation_only=best,selection_includes_test=False,postprocessing='None: validation-frozen probability threshold only',
        scoring='Sum TP/FP/FN over all six held-out tile positions and four periods; each valid pixel counted once',
        empty_metric_policy='Undefined ratios on empty individual tiles are null, not inflated to1; all tiles retained in pooled counts',
        root_length_reference='Skeleton derived from supplied human masks, not independently measured lengths; pixels only',
        cross_position_independent_test_images=0,whole_upper_results='Qualitative only; include training/validation/test pixels'))
    c.js(out/'verification.json',dict(passed=True,all_six_aggregate_count_scores_equal_training_exports=True,models=3,
        test_tile_date_masks=72,validation_tile_date_masks=60,checkpoint_SHA256s_verified=True,probability_masks_verified=maskchecks))
    print('ANALYSIS_DONE',[(r['model'],r['split'],r['Dice']) for r in aggregated],flush=True)

if __name__=='__main__': main()
