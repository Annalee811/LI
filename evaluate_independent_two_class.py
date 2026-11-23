"""Locked cross-tube two-class evaluation; missing/empty human GT never scores."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parent
def metrics(cm):
 classes=[]
 for k,name in [(1,'existing'),(2,'new')]:
  tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp)
  classes.append(dict(class_name=name,TP=tp,FP=fp,FN=fn,Precision=tp/(tp+fp)if tp+fp else None,Recall=tp/(tp+fn)if tp+fn else None,
   Dice=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,IoU=tp/(tp+fp+fn)if tp+fp+fn else None))
 return dict(classes=classes,macro_Dice=float(np.mean([r['Dice']for r in classes if r['Dice']is not None]))if any(r['Dice']is not None for r in classes)else None,
  macro_IoU=float(np.mean([r['IoU']for r in classes if r['IoU']is not None]))if any(r['IoU']is not None for r in classes)else None,confusion_background_existing_new=cm.tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--prediction-dir',type=Path,required=True);ap.add_argument('--annotation-root',type=Path);ap.add_argument('--gt-encoding',choices=['grouped-two','original-three'],default='grouped-two');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists()and any(a.output.iterdir()):raise ValueError('Choose a new empty output directory.')
 a.output.mkdir(parents=True,exist_ok=True)
 with a.manifest.open(encoding='utf-8-sig')as f:rows=[r for r in csv.DictReader(f)if r['role']=='locked_independent_test']
 keys=['position_id','tube_id','period','RGB_SHA256'];lock=json.loads((W/'Independent_locked_protocol.json').read_text())['locked_observations']
 assert sorted([{k:r[k]for k in keys}for r in rows],key=str)==sorted(lock,key=str),'Keep all locked observations; do not select a scored subset'
 assert all(r['tube_id']!='0353-2'and r['position_id']not in ['Q1-1-Z2','0353-1-Z2','Q1-3-Z3']for r in rows)
 scored=[r for r in rows if r['previous_observed_period']];missing=[];valid=[]
 schema='background_0_existing_1_new_2'if a.gt_encoding=='grouped-two'else'background_0_persistent_1_new_2_extension_3'
 for r in scored:
  p=a.annotation_root/r['position_id']/(r['period']+'_GT.png')if a.annotation_root else Path(r['GT_path'])
  if r.get('annotation_state')!='human_complete'or not r.get('annotator')or not r.get('annotation_source')or r.get('annotation_schema')!=schema or not p.is_file():
   missing.append(dict(position=r['position_id'],period=r['period'],reason='Human-complete two-group/three-group GT provenance and explicit label schema required'));continue
  gt=np.array(Image.open(p));assert gt.ndim==2 and gt.shape==(int(r['height']),int(r['width']))
  assert set(np.unique(gt)).issubset({0,1,2,255}if a.gt_encoding=='grouped-two'else{0,1,2,3,255})
  if not(gt!=255).any():missing.append(dict(position=r['position_id'],period=r['period'],reason='All-255 empty template is not ground truth'));continue
  if a.gt_encoding=='original-three':gt=gt.copy();gt[gt==3]=1
  valid.append((r,p,gt))
 if missing:
  status=dict(status='NOT_EVALUATED_NO_HUMAN_GT',classes={'existing':None,'new':None},macro_Dice=None,macro_IoU=None,input_frames=len(rows),baseline_excluded=len(rows)-len(scored),temporal_frames=len(scored),missing_GT=missing)
  (a.output/'Independent_status.json').write_text(json.dumps(status,indent=2));print('NOT_EVALUATED_NO_HUMAN_GT',len(missing));raise SystemExit(3)
 pooled=np.zeros((3,3),np.int64);tubes={};perframe=[];gt_hashes={}
 for r,p,gt in valid:
  pred=np.array(Image.open(a.prediction_dir/r['position_id']/(r['period']+'_two_classes.png')));assert pred.ndim==2 and pred.shape==gt.shape
  assert set(np.unique(pred)).issubset({0,1,2}),'Use raw grouped predictions; rejecting uncertainty pixels must not hide errors'
  v=gt!=255;cm=np.bincount(gt[v].astype(int)*3+pred[v].astype(int),minlength=9).reshape(3,3);pooled+=cm;tubes.setdefault(r['tube_id'],np.zeros((3,3),np.int64));tubes[r['tube_id']]+=cm
  perframe.append(dict(position=r['position_id'],period=r['period'],human_valid_pixels=int(v.sum()),metrics=metrics(cm)));gt_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 per_tube={t:metrics(c)for t,c in tubes.items()}
 tube_macro={}
 for key in ['macro_Dice','macro_IoU']:
  x=[m[key]for m in per_tube.values()if m[key]is not None]
  tube_macro[key]=dict(mean=float(np.mean(x))if x else None,SD=float(np.std(x,ddof=1))if len(x)>1 else None,n_tubes=len(x))
 result=dict(status='EVALUATED',domain='All human-valid pixels including background, T2–T4; two root classes scored',pooled=metrics(pooled),per_tube=per_tube,tube_macro=tube_macro,per_frame=perframe,GT_SHA256=gt_hashes,prior_unlabelled_image_exposure_documented=True)
 (a.output/'Independent_two_class_results.json').write_text(json.dumps(result,indent=2));print('EVALUATED_TWO_CLASS',len(tubes),'tubes',len(valid),'temporal frames')
if __name__=='__main__':main()
