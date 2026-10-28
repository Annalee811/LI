"""Full maps with a strict candidate edit gate; audit every unchanged outer pixel."""
import json
import numpy as np
import torch
from PIL import Image
from data import Data
from model import LocalCorrector
from train import proposed,metrics,edit_stats
from prepare import ROOT,OUT,ANCHOR,BASE,PERIODS,js,sha,csvout
from metrics import binary_curve

def link_metric(y,p,th):
 q=p>=th;tp=int((q&(y==1)).sum());fp=int((q&(y==0)).sum());fn=int((~q&(y==1)).sum());r=binary_curve(y==1,p);return dict(Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1),AP=r['average_precision'],AUC=r['AUC'],threshold=float(th),positive_pairs=int((y==1).sum()),negative_pairs=int((y==0).sum()))
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);data=Data();done=json.loads((OUT/'completion.json').read_text());assert done['epochs_completed']==100;assert sha(OUT/'best.pt')==done['checkpoint_sha256'];c=torch.load(OUT/'best.pt',weights_only=False,map_location='cpu');model=LocalCorrector();model.load_state_dict(c['state_dict']);model.eval();h=torch.from_numpy(np.load(OUT/'inference.npz')['h']);th=done['edit_threshold'];(OUT/'full').mkdir(exist_ok=True);(OUT/'changes').mkdir(exist_ok=True);audit=[];allbefore=[];allafter=[];alltarget=[];oldpred=[];localtarget=[];locold=[];locafter=[];maps={};identities=[]
 for ti,t in enumerate(PERIODS):
  original=np.array(Image.open(OUT/'frozen_unet'/f'{t}_classes.png'));p=dict(np.load(OUT/'data/unet'/(t+'_local.npz')));candidate=np.load(OUT/'candidates'/f'{t}.npz')['eligible'];after=original.copy()
  if len(p['node']):
   tensor={k:torch.from_numpy(v) for k,v in p.items()}
   with torch.inference_mode():q,confidence,allow,_,_=proposed(model,h,tensor)
   edit=(confidence>=th)&allow;values=p['anchor'].copy();values[edit]=q[edit];after.ravel()[p['pixel']]=values+1
  changed=after!=original;assert not changed[~candidate].any(),'Unauthorized body-class edit';assert np.array_equal(after>0,original>0),'Geometry changed';assert not ti or np.isin(after[original>0],[1,2,3]).all();assert ti or np.array_equal(original,after)
  Image.fromarray(after).save(OUT/'full'/f'{t}_classes.png');Image.fromarray(changed.astype(np.uint8)*255).save(OUT/'changes'/f'{t}_edited_pixels.png');maps[t]=(original,after)
  native=dict(np.load(OUT/'data/unet'/(t+'_pixels.npz')));use=native['split']==2
  if ti:
   alltarget.extend(native['target'][use]);allbefore.extend(original.ravel()[native['pixel'][use]]-1);allafter.extend(after.ravel()[native['pixel'][use]]-1);old=np.array(Image.open(BASE/'unet/full'/f'{t}_classes.png'));oldpred.extend(old.ravel()[native['pixel'][use]]-1)
   for rid in np.unique(native['rid'][use]):
    if rid<0:continue
    take=use&(native['rid']==rid);b=original.ravel()[native['pixel'][take]];a=after.ravel()[native['pixel'][take]]
    if len(a)<10:continue
    identities.append(dict(period=t,root_target=int(rid),pixels=len(a),before_new_extension_conflict_5px=bool(min((b==2).sum(),(b==3).sum())>=5),after_new_extension_conflict_5px=bool(min((a==2).sum(),(a==3).sum())>=5)))
   use2=p['split']==2;localtarget.extend(p['target'][use2]);locold.extend(p['anchor'][use2]);locafter.extend(after.ravel()[p['pixel'][use2]]-1)
  audit.append(dict(period=t,root_pixels=int((original>0).sum()),eligible_pixels=int(candidate.sum()),edited_pixels=int(changed.sum()),outside_candidate_edited_pixels=int(changed[~candidate].sum()),frozen_anchor_sha256=sha(OUT/'frozen_unet'/f'{t}_classes.png'),output_sha256=sha(OUT/'full'/f'{t}_classes.png'),binary_identical=True))
 y=np.array(alltarget,np.int64);measure={name:metrics(y,np.array(p,np.int64)) for name,p in [('original_U_Net_V75_before',allbefore),('original_U_Net_plus_latest_local_CG',allafter)]};rows=[]
 for variant,m in measure.items():
  for row in m['classes']:rows.append(dict(variant=variant,scope='fixed_root_internal_spatial_holdout',**row))
 csvout(OUT/'Three_temporal_classes_before_after.csv',rows);csvout(OUT/'Pixel_change_audit.csv',audit)
 if identities:csvout(OUT/'Known_root_identity_conflicts.csv',identities)
 lp=np.array(locafter,np.int64);lt=np.array(localtarget,np.int64);la=np.array(locold,np.int64);localstats=edit_stats(dict(target=lt,anchor=la),lp);assert localstats==done['test_local_edits'];historical={};compare_rows=[];spatial=np.load(OUT/'data/spatial.npy')
 for ref_version in ['20','24']:
  all_y=[];bb=[];aa=[]
  root=ROOT/'results/B3_ConvLSTM_GAT_upload20_v74_20261028/data/reference' if ref_version=='20' else OUT/'data/reference'
  for t in PERIODS[1:]:
   gt=np.array(Image.open(root/f'{t}_classes.png'));b,a=maps[t];take=(spatial==2)&(b>0)&np.isin(gt,[1,2,3]);all_y.extend(gt[take].astype(np.int64)-1);bb.extend(b[take].astype(np.int64)-1);aa.extend(a[take].astype(np.int64)-1)
  yy=np.array(all_y,np.int64);before_metric=metrics(yy,np.array(bb,np.int64));after_metric=metrics(yy,np.array(aa,np.int64));historical[ref_version]=dict(before=before_metric,after=after_metric)
  for variant,metric in [('original_U_Net',before_metric),('original_U_Net_plus_latest_local_CG',after_metric)]:
   for row in metric['classes']:compare_rows.append(dict(reference=ref_version,variant=variant,**row))
 expected=json.loads((ROOT/'results/B3_ConvLSTM_GAT_UNet_preview_v75_20261028/metrics.json').read_text())['original']['confusion'];assert historical['20']['before']['confusion']==expected,'Historical original baseline must exactly match .8229 comparison'
 assert historical['24']['before']['confusion']==measure['original_U_Net_V75_before']['confusion'];csvout(OUT/'Reference20_vs24_paired_comparison.csv',compare_rows)
 js(OUT/'metrics.json',dict(whole=measure,local_before=metrics(lt,la),local_after=metrics(lt,lp),test_local_edit_stats=localstats,paired_reference_evaluations=historical))
 cached=np.load(OUT/'inference.npz');conn=cached['connection'];linkrows=[]
 for same,name in [(True,'within_period'),(False,'across_periods')]:
  vi=np.flatnonzero((data.g['split']==1)&(data.same==same));tt=np.flatnonzero((data.g['split']==2)&(data.same==same));target=data.g['target'];threshold=max(np.linspace(.1,.95,35),key=lambda k:(lambda r:r['F1'] if r['Precision']>=.8 else -.1)(link_metric(target[vi],conn[vi],k)));linkrows.append(dict(scope=name,**link_metric(target[tt],conn[tt],threshold)))
 csvout(OUT/'Root_connection_metrics.csv',linkrows)
 # Difference illustrations selected by actual changed pixel counts, not favorable accuracy.
 score=np.zeros((1108,2541),np.int32)
 for t,(a,b) in maps.items():score+=(a!=b)
 choices=[]
 for yy in range(0,1108-256+1,64):
  for xx in range(0,2541-256+1,64):choices.append((int(score[yy:yy+256,xx:xx+256].sum()),[xx,yy,xx+256,yy+256]))
 choices.sort(key=lambda v:(-v[0],v[1]));rois=[]
 for value,r in choices:
  if any(max(0,min(r[2],o['ROI'][2])-max(r[0],o['ROI'][0]))*max(0,min(r[3],o['ROI'][3])-max(r[1],o['ROI'][1]))>256**2*.15 for o in rois):continue
  rois.append(dict(ROI=r,changed_pixels=value))
  if len(rois)==3:break
 js(OUT/'ROI_selection.json',dict(windows=rois,selection='Actual edit counts only, no accuracy input; same window across four periods; examples may include development pixels'))
 prep=json.loads((OUT/'preparation.json').read_text())
 for row in prep['immutable_inputs']:assert sha(row['path'])==row['sha256']
 js(OUT/'evaluation_verification.json',dict(passed=True,source_sha256=prep['source_sha256'],reference='24',epochs_local=100,anchor_epochs=json.loads((ANCHOR/'unet/completion.json').read_text())['epochs_completed'],all_binary_masks_identical=True,all_body_class_changes_outside_candidates_zero=True,T1_unchanged=True,heldout_predictions_match_training=True,reference_IDs_not_prediction_inputs=True,changed_pixels=sum(r['edited_pixels'] for r in audit),unchanged_inputs=True))
 print('VERIFIED LOCAL ONLY',localstats,'edited',sum(r['edited_pixels'] for r in audit),flush=True)
if __name__=='__main__':main()
