"""Aggregate both reference versions without substituting model agreement for truth."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'tables';out.mkdir(exist_ok=True);aggregate=[];by_period=[];beforeafter=[];length_pairs=[];length_summary=[];all_metrics={};manifest=c.rows(run/'data/paired_manifest.csv');lookup={r['pair_id']:r for r in manifest}
for model in c.MODELS:
 folder=run/model;d=json.loads((folder/'metrics.json').read_text());all_metrics[model]=d
 for split in ['val','test']:
  for ref in ['original','revised']:
   for kind in ['end_to_end','conditional_on_detected_reference_roots']:
    for row in d[split][ref][kind]['classes']:aggregate.append(dict(model=model,split=split,reference=ref,metric_scope=kind,macro_Dice=d[split][ref][kind]['macro_Dice'],**row))

  rows=c.rows(folder/'evaluation'/split/'per_pair_class_metrics.csv')
  for ref in ['original','revised']:
   for t in c.PERIODS[1:]:
    for k in [1,2,3]:
     rr=[r for r in rows if r['reference']==ref and r['current']==t and int(r['class_id'])==k];tp=sum(int(r['TP']) for r in rr);fp=sum(int(r['FP']) for r in rr);fn=sum(int(r['FN']) for r in rr)
     by_period.append(dict(model=model,split=split,reference=ref,current=t,class_id=k,class_name=c.NAMES[k],TP=tp,FP=fp,FN=fn,Precision=tp/(tp+fp) if tp+fp else None,Recall=tp/(tp+fn) if tp+fn else None,Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,IoU=tp/(tp+fp+fn) if tp+fp+fn else None))
 for row in [r for r in manifest if r['split']=='test']:
  pred=np.array(Image.open(folder/'evaluation/test'/(row['pair_id']+'_prediction.png')));x,y=int(row['x']),int(row['y'])
  for ref in ['original','revised']:
   gt=np.array(Image.open(run/'data'/ref/(row['current']+'_classes.png')))[y:y+256,x:x+256];valid=gt!=255
   for k in [0,1,2,3]:
    gm=((gt>0)&(gt<255)) if k==0 else gt==k;pm=((pred>0)&valid) if k==0 else (pred==k)&valid
    length_pairs.append(dict(model=model,reference=ref,pair_id=row['pair_id'],current=row['current'],class_id=k,class_name='all_typed_roots' if k==0 else c.NAMES[k],reference_skeleton_pixels=int(skeletonize(gm).sum()),predicted_skeleton_pixels=int(skeletonize(pm).sum())))
for model in c.MODELS:
 for ref in ['original','revised']:
  for k in [0,1,2,3]:
   rr=[r for r in length_pairs if r['model']==model and r['reference']==ref and r['class_id']==k];x=np.array([r['reference_skeleton_pixels'] for r in rr],float);y=np.array([r['predicted_skeleton_pixels'] for r in rr],float);error=y-x;sst=((x-x.mean())**2).sum();identity=1-(error**2).sum()/sst if sst>0 else None
   if len(x)>1 and np.var(x)>0 and np.var(y)>0:slope,intercept=np.polyfit(x,y,1);r2=float(np.corrcoef(x,y)[0,1]**2)
   else:slope=intercept=r2=None
   den=np.var(x)+np.var(y)+(x.mean()-y.mean())**2;ccc=float(2*np.mean((x-x.mean())*(y-y.mean()))/den) if den>0 else None
   length_summary.append(dict(model=model,reference=ref,class_id=k,n_correlated_tile_date_pairs=len(rr),R2_identity=float(identity) if identity is not None else None,R2_OLS=r2,RMSE_skeleton_pixels=float(np.sqrt((error**2).mean())),MAE_skeleton_pixels=float(np.abs(error).mean()),bias_skeleton_pixels=float(error.mean()),CCC=ccc,slope=float(slope) if slope is not None else None,intercept=float(intercept) if intercept is not None else None))
c.csvout(out/'class_metrics.csv',aggregate);c.csvout(out/'per_transition_metrics.csv',by_period);c.csvout(out/'skeleton_length_pairs.csv',length_pairs);c.csvout(out/'skeleton_length_agreement.csv',length_summary);c.js(run/'all_metrics.json',all_metrics)
c.js(out/'report_protocol.json',dict(references={'original':'Original human temporal/geometry labels, same-field development audit','revised':'Human temporal classes preserved; U-Net morphology and untyped added roots'},checkpoint_selection='Manual-temporal/morphology-reference full-frame validation whole-axis macroDice every4epochs; decoder strength on same validation windows',root_stage_frozen=True,temporal_heads_all_trained_32_epochs=True,selection_includes_epoch_zero=True,validation_pairs=15,internal_test_pairs=18,independent_external_test_images=0,historical_same_field_warmstart_exposure_possible=True,U_Net_temporal_teacher_used=False,original_human_preserved_separately=True,whole_axis_prediction_context_crosses_split_windows=True,no_independent_manual_length_measurement=True,length_units='skeleton pixel counts, not calibrated physical root lengths',full_upper_maps_include_training_pixels=True,root_frameworks={'unet':'PyTorch','deeplab':'TensorFlow1.12','segformer':'PyTorch'},temporal_framework='PyTorch',T1_baseline_not_inflated_into_temporal_macro_Dice=True))
print('REPORT_TABLES_READY',flush=True)
