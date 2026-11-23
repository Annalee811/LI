from pathlib import Path
import json,csv
import numpy as np
W=Path(__file__).resolve().parents[1];protocol=json.loads((W/'reports/locked_protocol.json').read_text());results=[]
for variant in protocol['models']:
 for seed in protocol['seeds']:
  p=W/'experiments'/variant/str(seed)/'result.json'
  if not p.exists():continue
  r=json.loads(p.read_text());assert r['epochs_completed']==100;results.append(r)
def csvout(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
perrun=[];perclass=[]
for r in results:
 t=r['test'];perrun.append(dict(variant=r['variant'],seed=r['seed'],epochs=r['epochs_completed'],selected_epoch=r['selected_epoch'],parameters=r['parameters'],test_macro_Dice=t['macro_Dice'],test_macro_IoU=t['macro_IoU'],test_identity_link_F1=t['identity_link_F1'],test_junction_assignment=t['junction_assignment']['accuracy'],test_junction_events=t['junction_assignment']['total'],test_identity_links=t['identity_links'],scope='B3 internal feature study; historical upstream exposure'))
 for c in t['classes']:perclass.append(dict(variant=r['variant'],seed=r['seed'],class_name=c['name'],support=c['support'],Precision=c['Precision'],Recall=c['Recall'],Dice=c['Dice'],IoU=c['IoU']))
if perrun:csvout(W/'reports/Internal_results_per_run.csv',perrun);csvout(W/'reports/Internal_results_per_class.csv',perclass)
summary=[]
for variant in protocol['models']:
 rr=[r for r in perrun if r['variant']==variant]
 if not rr:continue
 out=dict(variant=variant,seeds=len(rr),parameters=rr[0]['parameters'])
 for key in ['test_macro_Dice','test_macro_IoU','test_identity_link_F1','test_junction_assignment']:
  vals=np.array([r[key]for r in rr]);out[key+'_mean']=float(vals.mean());out[key+'_SD']=float(vals.std(ddof=1))if len(vals)>1 else None
 for k in ['persistent','new','extension']:
  vals=np.array([r['Dice']for r in perclass if r['variant']==variant and r['class_name']==k]);out[k+'_Dice_mean']=float(vals.mean());out[k+'_Dice_SD']=float(vals.std(ddof=1))if len(vals)>1 else None
 summary.append(out)
if summary:csvout(W/'reports/Internal_results_mean_SD.csv',summary)
# All frozen references use the same eligible pixels but are not part of the matched fresh training comparison.
frozen=json.loads((W/'reports/frozen_references.json').read_text());ref=[]
for name,r in frozen.items():
 for c in r['metrics_by_split']['2']['classes']:ref.append(dict(reference=name,class_name=c['name'],Dice=c['Dice'],IoU=c['IoU'],Precision=c['Precision'],Recall=c['Recall'],scope=r['scope']))
csvout(W/'reports/Historical_reference_and_decoder_stage_scores.csv',ref)
# Quantify deterministic paired decoder-stage changes separately.
decrows=[]
for seed in protocol['seeds']:
 root=W/'decoder_experiments'/str(seed)
 if not(root/'completed.json').exists():continue
 from PIL import Image
 from train_study import metrics
 for stage,sub in [('raw_head','branch/full'),('axis_constraints','axis/full'),('axis_and_junction_constraints','junction/full')]:
  cm=np.zeros((3,3),np.int64)
  for t in ['T2','T3','T4']:
   p=np.load(W/f'data/{t}_pixels.npz');ok=p['split']==2;v=np.array(Image.open(root/sub/(t+'_classes.png'))).ravel()[p['pixel'][ok]]-1;assert np.isin(v,[0,1,2]).all();cm+=np.bincount(p['target'][ok]*3+v,minlength=9).reshape(3,3)
  m=metrics(cm);identity=None
  if stage!='raw_head':
   pairTP=pairPred=pairGT=0;idpixels=0
   for t in ['T2','T3','T4']:
    q=np.load(W/f'data/{t}_pixels.npz');ok=(q['split']==2)&(q['rid']>=0);own=np.load(root/('axis' if stage=='axis_constraints' else 'junction')/'ownership'/(t+'_axis_owner.npz'))['axis'].ravel()[q['pixel'][ok]];truth=q['rid'][ok];valid=own>=0;truth=truth[valid];own=own[valid];idpixels+=len(own);_,a=np.unique(truth,return_inverse=True);_,b=np.unique(own,return_inverse=True);n=int(b.max())+1 if len(b)else 1;ct=np.bincount(a*n+b,minlength=(int(a.max())+1)*n).reshape(-1,n) if len(a)else np.zeros((0,n),np.int64);pairTP+=int((ct*(ct-1)//2).sum());pr=ct.sum(0);gt=ct.sum(1);pairPred+=int((pr*(pr-1)//2).sum());pairGT+=int((gt*(gt-1)//2).sum())
   identity=dict(pairwise_instance_F1=2*pairTP/(pairPred+pairGT)if pairPred+pairGT else None,pairwise_instance_precision=pairTP/pairPred if pairPred else None,pairwise_instance_recall=pairTP/pairGT if pairGT else None,identity_evaluation_pixels=idpixels)
  row=dict(seed=seed,stage=stage,macro_Dice=m['macro_Dice'],macro_IoU=m['macro_IoU'],scope='Same whole-field neural outputs per seed; separate from partition-isolated primary experiments')
  row.update(identity if identity is not None else dict(pairwise_instance_F1=None,pairwise_instance_precision=None,pairwise_instance_recall=None,identity_evaluation_pixels=None))
  for c in m['classes']:row[c['name']+'_Dice']=c['Dice']
  decrows.append(row)
if decrows:csvout(W/'reports/Paired_decoder_stage_results.csv',decrows)
(W/'reports/summary_status.json').write_text(json.dumps(dict(formal_runs_completed=len(results),formal_runs_expected=15,decoder_seed_stage_rows=len(decrows),independent_test_status='NOT_EVALUATED_NO_HUMAN_GT',independent_test_tubes=3,independent_test_frames=26,seed_SD_is_optimization_variation_not_biological_replication=True),indent=2));print('SUMMARIZED',len(results),'training runs;',len(decrows),'decoder rows')
