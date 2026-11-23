"""Newly observed class only: existing outputs, unchanged human truth and models.

Positive = original temporal class 2. All other valid classes are negatives.
Nothing is merged into class 2. Only Precision and Recall are reported.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parent
STUDY=Path('/workspace/work/V90_validation_20261123')
OLD=Path('/workspace/work/V90_all_microtubes_20261118')
REF=Path('/workspace/work/V66_V90_same_field_SCI_20261113/sources/reference')
def csvout(path,rows):
 path.parent.mkdir(exist_ok=True,parents=True)
 with path.open('w',encoding='utf-8-sig',newline='')as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def pr(tp,fp,fn):
 return dict(TP=tp,FP=fp,FN=fn,new_truth_pixels=tp+fn,new_predicted_pixels=tp+fp,
  Precision=tp/(tp+fp)if tp+fp else None,Recall=tp/(tp+fn)if tp+fn else None)
def score(folder,periods=('T2','T3','T4'),domain='heldout'):
 tp=fp=fn=0
 for t in periods:
  pred=np.array(Image.open(folder/(t+'_classes.png')))
  if domain=='heldout':
   q=np.load(STUDY/'data'/(t+'_pixels.npz'));v=q['split']==2
   y=q['target'][v]==1;p=pred.ravel()[q['pixel'][v]]==2
  else:
   gt=np.array(Image.open(REF/(t+'_classes.png')));v=gt!=255
   y=gt[v]==2;p=pred[v]==2
  tp+=int((y&p).sum());fp+=int((~y&p).sum());fn+=int((y&~p).sum())
 return pr(tp,fp,fn)
def main():
 R=W/'metrics';R.mkdir(exist_ok=True)
 protocol=json.loads((STUDY/'reports/locked_protocol.json').read_text())
 runs=[];summary=[]
 for variant in protocol['models']:
  for seed in protocol['seeds']:
   r=dict(variant=variant,seed=seed,scope='Internal B3 held-out eligible typed root pixels; frozen foreground; historical upstream exposure',**score(STUDY/'experiments'/variant/str(seed)/'predictions'))
   runs.append(r)
  rr=[r for r in runs if r['variant']==variant];s=dict(variant=variant,seeds=len(rr),new_truth_pixels=rr[0]['new_truth_pixels'])
  for k in ['Precision','Recall']:
   x=[r[k]for r in rr if r[k]is not None];s[k+'_mean']=float(np.mean(x))if x else None;s[k+'_SD']=float(np.std(x,ddof=1))if len(x)>1 else None;s[k+'_defined_seeds']=len(x)
  summary.append(s)
 csvout(R/'New_root_internal_each_run.csv',runs);csvout(R/'New_root_internal_mean_SD.csv',summary)
 dec=[]
 for seed in protocol['seeds']:
  for stage,sub in [('raw_head','branch/full'),('axis_constraints','axis/full'),('axis_and_junction_constraints','junction/full')]:
   dec.append(dict(seed=seed,stage=stage,scope='Separate paired whole-field context; same neural output within seed; eligible held-out typed root pixels',**score(STUDY/'decoder_experiments'/str(seed)/sub)))
 csvout(R/'New_root_paired_decoder.csv',dec)
 frozen=[];historical=[]
 for name,folder in [('V66_frozen',OLD/'model_source/assets/base/unet/full'),('V90_automatic_historical',OLD/'B3_verified_inference/junction/full')]:
  frozen.append(dict(model=name,scope='Historical frozen reference on held-out eligible typed root pixels; no retraining',**score(folder)))
  for period in ['T2','T3','T4','T2-T4_pooled']:
   historical.append(dict(model=name,period=period,scope='Full B3 image human-valid pixels including background; descriptive historical agreement, not held-out or independent accuracy',**score(folder,periods=(period,)if period!='T2-T4_pooled'else('T2','T3','T4'),domain='full')))
 csvout(R/'New_root_frozen_heldout_reference.csv',frozen);csvout(R/'New_root_B3_full_image_historical_agreement.csv',historical)
 # Compare to archived class-specific metrics; do not revise old outputs.
 original=list(csv.DictReader((STUDY/'reports/Internal_results_per_class.csv').open(encoding='utf-8-sig')))
 for r in runs:
  old=next(x for x in original if x['variant']==r['variant']and int(x['seed'])==r['seed']and x['class_name']=='new')
  for key in ['Precision','Recall']:
   assert (r[key]is None and old[key]=='')or abs(r[key]-float(old[key]))<1e-12
 summary_json={s['variant']:s for s in summary}
 decoder_summary={}
 for stage in ['raw_head','axis_constraints','axis_and_junction_constraints']:
  rr=[r for r in dec if r['stage']==stage];decoder_summary[stage]={}
  for k in ['Precision','Recall']:
   x=np.array([r[k]for r in rr]);decoder_summary[stage][k]=dict(mean=float(x.mean()),SD=float(x.std(ddof=1)),seeds=len(x))
 audit=dict(class_code_positive=2,other_valid_classes_remain_negatives=True,classes_not_merged=True,
  reported_metrics=['Precision','Recall'],new_predictions_generated=False,models_retrained=False,
  scope_changed_to_single_class_reporting=True,original_evaluation_masks_unchanged=True,
  current_frame_Precision_is_undefined_when_no_positive_predictions=True,
  internal_scores_match_archived_new_class_scores=True,independent_accuracy_available=False,
  GT_SHA256={t:hashlib.sha256((REF/(t+'_classes.png')).read_bytes()).hexdigest()for t in ['T1','T2','T3','T4']})
 (R/'New_root_summary.json').write_text(json.dumps(dict(internal=summary_json,paired_decoder=decoder_summary,audit=audit),indent=2))
 print('NEW_ONLY_PRECISION_RECALL_COMPLETE',len(runs),'runs;',len(dec),'decoder stages;',len(historical),'historical rows',flush=True)
if __name__=='__main__':main()
