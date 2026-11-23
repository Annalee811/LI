"""Two-class evaluation of existing checkpoints: Existing=P+E, New=N.

Confusion matrices are recalculated after merging truth and predictions, never
by adding or averaging old class scores. No checkpoint is retrained/reselected.
"""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parent
S=Path('/workspace/work/V90_validation_20261123')
O=Path('/workspace/work/V90_all_microtubes_20261118')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='')as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def metrics(cm):
 result=[]
 for i,name in enumerate(['existing','new']):
  tp=int(cm[i,i]);fp=int(cm[:,i].sum()-tp);fn=int(cm[i].sum()-tp)
  result.append(dict(class_name=name,TP=tp,FP=fp,FN=fn,support=tp+fn,Precision=tp/(tp+fp)if tp+fp else None,
   Recall=tp/(tp+fn)if tp+fn else None,Dice=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,IoU=tp/(tp+fp+fn)if tp+fp+fn else None))
 return dict(classes=result,macro_Dice=float(np.mean([r['Dice']for r in result])),macro_IoU=float(np.mean([r['IoU']for r in result])),confusion=cm.tolist())
def evaluate(folder,export):
 cm=np.zeros((2,2),np.int64);old=np.zeros((3,3),np.int64);export.mkdir(parents=True,exist_ok=True)
 for t in ['T1','T2','T3','T4']:
  p=folder/(t+'_classes.png');raw=np.array(Image.open(p));mapped=np.zeros(raw.shape,np.uint8);mapped[np.isin(raw,[1,3])]=1;mapped[raw==2]=2;mapped[np.isin(raw,[4,255])]=255
  Image.fromarray(mapped).save(export/(t+'_two_classes.png'))
  if t=='T1':continue
  q=np.load(S/'data'/(t+'_pixels.npz'));v=q['split']==2;y=q['target'][v];p=raw.ravel()[q['pixel'][v]];assert np.isin(p,[1,2,3]).all()
  gt_new=(y==1).astype(np.int64);pred_new=(p==2).astype(np.int64)
  cm+=np.bincount(gt_new*2+pred_new,minlength=4).reshape(2,2)
  old+=np.bincount(y*3+(p-1),minlength=9).reshape(3,3)
 membership=np.array([[1,0],[0,1],[1,0]],np.int64);assert np.array_equal(cm,membership.T@old@membership)
 return metrics(cm)
def main():
 (W/'metrics').mkdir(exist_ok=True);data=W/'reproduction';data.mkdir(exist_ok=True)
 protocol=json.loads((S/'reports/locked_protocol.json').read_text());runs=[];classrows=[];details={};truth={}
 for t in ['T2','T3','T4']:
  q=np.load(S/'data'/(t+'_pixels.npz'));v=q['split']==2;truth[t+'_pixel']=q['pixel'][v];truth[t+'_truth']=(q['target'][v]==1).astype(np.uint8)
 np.savez_compressed(data/'heldout_truth.npz',**truth)
 checkpoints={}
 for variant in protocol['models']:
  for seed in protocol['seeds']:
   source=S/'experiments'/variant/str(seed);m=evaluate(source/'predictions',data/'internal'/variant/str(seed));details[variant+'/'+str(seed)]=m
   ck=json.loads((source/'result.json').read_text());checkpoints[variant+'/'+str(seed)]=dict(SHA256=sha(source/'best.pt'),selected_epoch=ck['selected_epoch'],selection='Original three-class validation criterion; unchanged')
   runs.append(dict(variant=variant,seed=seed,macro_Dice=m['macro_Dice'],macro_IoU=m['macro_IoU'],pixels=int(np.array(m['confusion']).sum())))
   for c in m['classes']:classrows.append(dict(variant=variant,seed=seed,**c))
 # Historical frozen V66 is one reference, not a third random-seed replicate.
 refs=[];refdetails={}
 for name,folder in [('V66_pairwise_frozen',O/'model_source/assets/base/unet/full'),('V90_automatic_historical',O/'B3_verified_inference/junction/full')]:
  m=evaluate(folder,data/'historical'/name);refdetails[name]=m
  for c in m['classes']:refs.append(dict(model=name,scope='Frozen historical B3 reference, not matched fresh training',**c,macro_Dice=m['macro_Dice'],macro_IoU=m['macro_IoU']))
 dec=[];decclasses=[];decdetails={}
 for seed in protocol['seeds']:
  for stage,sub in [('raw_head','branch/full'),('axis_constraints','axis/full'),('axis_and_junction_constraints','junction/full')]:
   m=evaluate(S/'decoder_experiments'/str(seed)/sub,data/'decoder'/str(seed)/stage);decdetails[str(seed)+'/'+stage]=m
   dec.append(dict(seed=seed,stage=stage,macro_Dice=m['macro_Dice'],macro_IoU=m['macro_IoU']))
   for c in m['classes']:decclasses.append(dict(seed=seed,stage=stage,**c))
 csvout(W/'metrics/Internal_two_class_runs.csv',runs);csvout(W/'metrics/Internal_two_class_classwise.csv',classrows)
 csvout(W/'metrics/Historical_two_class_references.csv',refs);csvout(W/'metrics/Paired_decoder_two_class.csv',dec);csvout(W/'metrics/Paired_decoder_two_class_classwise.csv',decclasses)
 summary=[]
 for variant in protocol['models']:
  rr=[r for r in runs if r['variant']==variant];out=dict(variant=variant,seeds=3)
  for k in ['macro_Dice','macro_IoU']:
   x=np.array([r[k]for r in rr]);out[k+'_mean']=float(x.mean());out[k+'_SD']=float(x.std(ddof=1))
  for c in ['existing','new']:
   for k in ['Precision','Recall','Dice','IoU']:
    x=[r[k]for r in classrows if r['variant']==variant and r['class_name']==c and r[k]is not None]
    out[c+'_'+k+'_mean']=float(np.mean(x))if x else None;out[c+'_'+k+'_SD']=float(np.std(x,ddof=1))if len(x)>1 else None
  summary.append(out)
 csvout(W/'metrics/Two_class_mean_SD.csv',summary)
 effects=[]
 for ablated,module in [('no_lstm','ConvLSTM'),('no_gat','GAT'),('no_gates','Learned identity gates')]:
  full=[next(r for r in runs if r['variant']=='full'and r['seed']==s)for s in protocol['seeds']]
  other=[next(r for r in runs if r['variant']==ablated and r['seed']==s)for s in protocol['seeds']]
  for s,f,r in zip(protocol['seeds'],full,other):effects.append(dict(module=module,ablated_variant=ablated,seed=s,delta_macro_Dice_pp=100*(f['macro_Dice']-r['macro_Dice']),delta_macro_IoU_pp=100*(f['macro_IoU']-r['macro_IoU'])))
 csvout(W/'metrics/Paired_module_two_class_effects.csv',effects)
 oldrows=list(csv.DictReader((S/'reports/Internal_results_per_class.csv').open(encoding='utf-8-sig')))
 for r in classrows:
  if r['class_name']!='new':continue
  old=next(x for x in oldrows if x['variant']==r['variant']and int(x['seed'])==r['seed']and x['class_name']=='new')
  for k in ['Precision','Recall','Dice','IoU']:assert(r[k]is None and old[k]=='')or abs(r[k]-float(old[k]))<1e-12
 shutil.copy2(S/'reports/independent_protocol.json',W/'Independent_locked_protocol.json');shutil.copy2(S/'reports/annotation_checklist.csv',W/'Independent_manifest.csv');shutil.copy2(S/'reports/locked_protocol.json',W/'Original_training_protocol.json')
 (W/'metrics/All_two_class_confusions.json').write_text(json.dumps(dict(internal=details,historical=refdetails,paired_decoder=decdetails),indent=2))
 (W/'Existing_checkpoints.json').write_text(json.dumps(checkpoints,indent=2))
 audit=dict(passed=True,classes={'existing':[1,3],'new':[2]},truth_and_prediction_masks_merged_before_metrics=True,
  existing_class_scores_not_averaged_from_component_scores=True,new_class_scores_unchanged=True,two_class_macro_not_comparable_to_three_class_macro=True,
  original_checkpoints_and_selection_unchanged=True,model_retrained=False,model_forward_rerun=False,
  internal_domain='83,478 common eligible held-out typed root pixels at T2–T4; foreground conditioned, background excluded',existing_truth_pixels=70907,new_truth_pixels=12571,
  primary_inference_context='Partition-isolated feature/module study',decoder_inference_context='Separate paired whole-field context',
  independent_human_GT_available=False,independent_metrics=None,training_runs_reused=15,optimization_updates_per_original_run=100)
 (W/'Two_class_evaluation_checks.json').write_text(json.dumps(audit,indent=2));print('TWO_CLASS_EXPERIMENT_REEVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
