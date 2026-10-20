"""Blend full-frame logits before one decision per RGB-traced root axis."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c

from structured import decode as structured

def evaluate(data,maps,split,out=None):
 cms={ref:np.zeros((4,4),np.int64) for ref in ['original','revised']};conds={ref:np.zeros((4,4),np.int64) for ref in cms};per=[]
 for row in [z for z in data.pairs if z['split']==split]:
  x,y=int(row['x']),int(row['y']);t=row['current'];pred=maps[t][y:y+256,x:x+256]
  if out:
   out.mkdir(parents=True,exist_ok=True);Image.fromarray(pred).save(out/(row['pair_id']+'_prediction.png'))
  for ref in cms:
   gt=data.targets[ref][t][y:y+256,x:x+256];cm=c.old.confusion(gt,pred);cms[ref]+=cm;cg=gt.copy();cg[~data.gate[t][y:y+256,x:x+256]]=255;conds[ref]+=c.old.confusion(cg,pred,True)
   for k in c.old.metrics(cm)['classes']:per.append(dict(pair_id=row['pair_id'],current=t,reference=ref,**k))
 result={ref:dict(end_to_end=c.old.metrics(cm),conditional_on_detected_reference_roots=c.old.metrics(conds[ref]),pair_count=sum(row['split']==split for row in data.pairs)) for ref,cm in cms.items()}
 if out:c.js(out/'metrics.json',result);c.csvout(out/'per_pair_class_metrics.csv',per)
 return result

def products(out,t,rgb,mask,classes,prob=None):
 out.mkdir(parents=True,exist_ok=True);Image.fromarray(rgb).save(out/(t+'_RGB.png'));Image.fromarray(classes).save(out/(t+'_temporal_classes.png'));Image.fromarray(mask.astype('uint8')*255).save(out/(t+'_root_mask.png'));Image.fromarray(c.PALETTE[classes]).save(out/(t+'_temporal_on_white.png'))
 overlay=rgb.copy();overlay[mask]=np.rint(.35*rgb[mask]+.65*c.PALETTE[classes[mask]]).astype('uint8');Image.fromarray(overlay).save(out/(t+'_temporal_overlay.png'))
 for suffix,im in [('original_roots_RGBA',rgb),('temporal_RGBA',c.PALETTE[classes])]:
  rgba=np.concatenate([im.copy(),mask[:,:,None].astype('uint8')*255],2);rgba[~mask,:3]=0;Image.fromarray(rgba).save(out/(t+'_'+suffix+'.png'))
 if prob is not None:
  np.savez_compressed(out/(t+'_class_probability.npz'),probability=(prob*mask[None]).astype('float16'));conf=np.take_along_axis(prob,np.clip(classes.astype(int)-1,0,2)[None],axis=0)[0]*mask;Image.fromarray(np.rint(conf*65535).astype('uint16')).save(out/(t+'_class_confidence_uint16.png'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args();torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');data=c.Data(a.run,a.roots,a.model);folder=a.run/a.model;model=c.load_head(folder/'best_temporal_head.pt',device);full=folder/'four_period_prediction';full.mkdir(exist_ok=True);probabilities={}
 for t in c.PERIODS[1:]:
  cache=full/(t+'_raw_probability.npz')
  if cache.exists():pr=np.load(cache)['probability'].astype('float32')
  else:pr=c.predict_probability(model,data.features[t],device);np.savez_compressed(cache,probability=pr.astype('float16'));pr=pr.astype('float16').astype('float32')
  probabilities[t]=pr
 # Only validation windows select the geometry/emission prior strength. No test
 # file is opened for this decision; geometry is label-free for the full image.
 candidates=[];best=-1.;selected=None;selected_maps=None;selected_records=None
 for strength in [0.,.5,1.,2.]:
  maps={};records={}
  for prev,t in zip(c.PERIODS,c.PERIODS[1:]):maps[t],records[t]=structured(probabilities[t],data.axes[prev],data.axes[t],data.link[t],strength)
  val=evaluate(data,maps,'val');score=val['revised']['end_to_end']['macro_Dice'];candidates.append(dict(strength=strength,validation_macro_Dice=score,original_validation_macro_Dice=val['original']['end_to_end']['macro_Dice']))
  if score>best+1e-8:best=score;selected=strength;selected_maps=maps;selected_records=records
  print('AXIS_DECODER_VALIDATION',a.model,strength,round(score,4),flush=True)
 c.js(folder/'decoder_selection.json',dict(selected_strength=selected,candidates=candidates,test_used_for_selection=False,whole_frame_logits_blended_before_axis_decision=True,semantics='All NEW, or terminal GROWTH / PERSISTENT / terminal GROWTH per traced root axis',crossings='Pair by opposite tangent, RGB appearance and width; unresolved flagged',axis_identity='Putative geometry, not independent biological identity'))
 products(full,'T1',data.rgb['T1'],data.gate['T1'],data.gate['T1'].astype('uint8')*4);Image.fromarray(data.axes['T1']['axis_map'].astype('uint16')).save(full/'T1_root_axis_ids_uint16.png');records=[]
 for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
  pred=selected_maps[t];products(full,t,data.rgb[t],data.gate[t],pred,probabilities[t]);c.js(full/(t+'_root_axes.json'),selected_records[t]);Image.fromarray(data.axes[t]['axis_map'].astype('uint16')).save(full/(t+'_root_axis_ids_uint16.png'));Image.fromarray(data.axes[t]['uncertain'].astype('uint8')*255).save(full/(t+'_crossing_review_mask.png'))
  # Explicitly measure whole-axis mixed NEW/GROWTH pixels after ownership.
  bad=0;raw_bad=0;raw_class=np.where(data.gate[t],probabilities[t].argmax(0)+1,0)
  for row in selected_records[t]:
   values=np.unique(pred[data.axes[t]['axis_map']==row['axis_id']]);bad+=int(2 in values and 3 in values);rv=np.unique(raw_class[data.axes[t]['axis_map']==row['axis_id']]);raw_bad+=int(2 in rv and 3 in rv)
  assert bad==0
  records.append(dict(period=t,previous=prev,axes=len(selected_records[t]),mixed_new_growth_axes=bad,mixed_new_growth_axes_before=raw_bad,unresolved_crossing_axes=sum(q['unresolved_crossing'] for q in selected_records[t]),class_pixels={str(k):int((pred==k).sum()) for k in [1,2,3]},RGB_SHA256=c.sha(a.run/'data/upper'/f'{t}_RGB.png')))
 reports={s:evaluate(data,selected_maps,s,folder/'evaluation'/s) for s in ['val','test']};c.js(folder/'metrics.json',reports);c.js(full/'inference.json',dict(version='temporal_reference_v16',head_SHA256=c.sha(folder/'best_temporal_head.pt'),manual_labels_used_as_model_or_axis_inputs=False,manual_labels_read_for_validation_and_evaluation=True,validation_labels_used_only_for_decoder_selection=True,root_stage_frozen=True,records=records,selected_strength=selected,confidence='Raw assigned-class network score; not a calibrated structured posterior',same_field_context='Whole axes can cross training and evaluation windows; no independent root/tube test'))
 # Retain an unconstrained counterfactual to quantify what grammar changed.
 raw={t:np.where(data.gate[t],probabilities[t].argmax(0)+1,0).astype('uint8') for t in c.PERIODS[1:]};c.js(folder/'raw_pixel_test_metrics.json',evaluate(data,raw,'test'))
 print('AXIS_PREDICTION_COMPLETE',a.model,'U_Net_reference_test_macro',reports['test']['revised']['end_to_end']['macro_Dice'],'original',reports['test']['original']['end_to_end']['macro_Dice'],flush=True)
if __name__=='__main__':main()
