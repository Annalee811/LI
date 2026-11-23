"""Human-GT-gated cross-tube evaluation. Pending/all-255 templates never produce accuracy numbers."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from PIL import Image

def class_metrics(cm):
 out=[]
 for k,name in [(1,'persistent'),(2,'new'),(3,'extension')]:
  tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp);out.append(dict(class_id=k,name=name,support=tp+fn,TP=tp,FP=fp,FN=fn,Precision=tp/(tp+fp)if tp+fp else None,Recall=tp/(tp+fn)if tp+fn else None,Dice=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,IoU=tp/(tp+fp+fn)if tp+fp+fn else None))
 return dict(classes=out,macro_Dice=float(np.mean([r['Dice']for r in out if r['Dice']is not None]))if any(r['Dice']is not None for r in out)else None,macro_IoU=float(np.mean([r['IoU']for r in out if r['IoU']is not None]))if any(r['IoU']is not None for r in out)else None,confusion=cm.tolist())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--prediction-dir',type=Path,required=True);ap.add_argument('--annotation-root',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=list(csv.DictReader(a.manifest.open(encoding='utf-8-sig')));rows=[r for r in rows if r.get('role')=='locked_independent_test'];assert rows,'No locked independent-test observations';locked=json.loads((Path(__file__).resolve().parents[1]/'reports/independent_protocol.json').read_text())['locked_observations'];actual=[{k:r[k]for k in ['position_id','tube_id','period','RGB_SHA256']}for r in rows];assert sorted(actual,key=str)==sorted(locked,key=str),'Formal test manifest differs from the locked cohort; do not select a scored subset';assert all(r['tube_id']!='0353-2'for r in rows),'B3 training tube cannot enter independent test';assert all(r['position_id']not in ['Q1-1-Z2','Q1-3-Z3','0353-1-Z2']for r in rows),'Excluded/legacy position in independent test';missing=[]
 for r in rows:
  p=a.annotation_root/r['position_id']/(r['period']+'_GT.png')if a.annotation_root else Path(r['GT_path'])
  if r.get('annotation_state')!='human_complete'or not r.get('annotator')or not r.get('annotation_source')or not p.exists():missing.append(dict(position=r['position_id'],period=r['period'],reason='Human-complete annotation/provenance missing'));continue
  gt=np.array(Image.open(p));assert gt.ndim==2 and gt.shape==(int(r['height']),int(r['width']));assert set(np.unique(gt)).issubset({0,1,2,3,4,255});
  if not(gt!=255).any():missing.append(dict(position=r['position_id'],period=r['period'],reason='All pixels unannotated; template is not ground truth'))
 if missing:
  (a.output/'evaluation_status.json').write_text(json.dumps(dict(status='NOT_EVALUATED_NO_HUMAN_GT',metrics=None,expected_frames=len(rows),missing_GT=missing),indent=2));print('NOT_EVALUATED_NO_HUMAN_GT',len(missing));raise SystemExit(3)
 pooled=np.zeros((4,4),np.int64);tubes={};perframe=[];gt_hashes={}
 for r in rows:
  gtpath=a.annotation_root/r['position_id']/(r['period']+'_GT.png')if a.annotation_root else Path(r['GT_path']);gt=np.array(Image.open(gtpath));predpath=a.prediction_dir/r['position_id']/(r['period']+'_classes.png');pred=np.array(Image.open(predpath));assert pred.shape==gt.shape and pred.ndim==2;assert set(np.unique(pred)).issubset({0,1,2,3,4}),'Use raw class prediction; uncertainty maps cannot silently hide hard pixels';valid=gt!=255;gp=(gt>0)&(gt<5);pp=(pred>0)&(pred<5);tp=int((gp&pp&valid).sum());fp=int((~gp&pp&valid).sum());fn=int((gp&~pp&valid).sum());cm=np.zeros((4,4),np.int64)
  if r['period']!='T1':
   assert not(gt[valid]==4).any()and not(pred[valid]==4).any(),'Baseline label outside first observation';cm+=np.bincount(gt[valid].astype(int)*4+pred[valid],minlength=16).reshape(4,4);pooled+=cm
  tube=r['tube_id'];tubes.setdefault(tube,dict(cm=np.zeros((4,4),np.int64),TP=0,FP=0,FN=0,frames=0));tubes[tube]['cm']+=cm;tubes[tube]['TP']+=tp;tubes[tube]['FP']+=fp;tubes[tube]['FN']+=fn;tubes[tube]['frames']+=1;perframe.append(dict(position=r['position_id'],tube=tube,period=r['period'],human_labeled_pixels=int(valid.sum()),root_Dice=2*tp/(2*tp+fp+fn)if 2*tp+fp+fn else None,temporal_class_metrics=class_metrics(cm)if r['period']!='T1'else None));gt_hashes[str(gtpath)]=hashlib.sha256(gtpath.read_bytes()).hexdigest()
 tubemetrics={}
 for tube,d in tubes.items():
  rr=class_metrics(d['cm']);rr.update(frames=d['frames'],root_Dice=2*d['TP']/(2*d['TP']+d['FP']+d['FN'])if 2*d['TP']+d['FP']+d['FN']else None);tubemetrics[tube]=rr
 result=dict(status='EVALUATED',scope='Cross-tube human-label-held-out evaluation; earlier unlabelled image/prediction exposure documented',temporal_domain='All human-valid image pixels including true background; T1 baseline excluded from temporal class scores',independent_tubes=len(tubes),frames=len(rows),tube_macro_Dice=float(np.mean([r['macro_Dice']for r in tubemetrics.values()if r['macro_Dice']is not None])),pooled=class_metrics(pooled),per_tube=tubemetrics,per_frame=perframe,GT_SHA256=gt_hashes,hyperparameter_selection_on_test=False)
 (a.output/'independent_results.json').write_text(json.dumps(result,indent=2));(a.output/'evaluation_status.json').write_text(json.dumps(dict(status='EVALUATED',independent_tubes=len(tubes),frames=len(rows)),indent=2));print('EVALUATED',len(tubes),'tubes',len(rows),'frames')
if __name__=='__main__':main()
