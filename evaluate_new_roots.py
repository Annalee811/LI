"""Human-GT-gated cross-tube Precision/Recall for newly observed roots only."""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parent
def stats(tp,fp,fn):
 return dict(TP=tp,FP=fp,FN=fn,Precision=tp/(tp+fp)if tp+fp else None,Recall=tp/(tp+fn)if tp+fn else None)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--manifest',type=Path,required=True)
 ap.add_argument('--prediction-dir',type=Path,required=True)
 ap.add_argument('--annotation-root',type=Path)
 ap.add_argument('--gt-encoding',choices=['temporal','new-binary'],default='temporal')
 ap.add_argument('--prediction-encoding',choices=['new-mask','temporal-class'],default='new-mask')
 ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args()
 if a.output.exists()and any(a.output.iterdir()):raise ValueError('Choose a new empty output directory.')
 a.output.mkdir(parents=True,exist_ok=True)
 with a.manifest.open(encoding='utf-8-sig')as f:rows=[r for r in csv.DictReader(f)if r['role']=='locked_independent_test']
 protocol=json.loads((W/'Independent_locked_protocol.json').read_text());keys=['position_id','tube_id','period','RGB_SHA256']
 assert sorted([{k:r[k]for k in keys}for r in rows],key=str)==sorted(protocol['locked_observations'],key=str),'Keep the entire locked 26-frame cohort'
 assert all(r['tube_id']!='0353-2'and r['position_id']not in ['Q1-1-Z2','0353-1-Z2','Q1-3-Z3']for r in rows)
 scored=[r for r in rows if r['previous_observed_period']];missing=[];validated=[]
 for r in scored:
  suffix='_GT.png'if a.gt_encoding=='temporal'else'_new_GT.png'
  p=a.annotation_root/r['position_id']/(r['period']+suffix)if a.annotation_root else Path(r['GT_path'])
  if r.get('annotation_state')!='human_complete'or not r.get('annotator')or not r.get('annotation_source')or not p.is_file():
   missing.append(dict(position=r['position_id'],period=r['period'],reason='Human-complete GT and provenance unavailable'));continue
  gt=np.array(Image.open(p));assert gt.shape==(int(r['height']),int(r['width']))and gt.ndim==2
  assert set(np.unique(gt)).issubset({0,1,2,3,255}if a.gt_encoding=='temporal'else{0,1,255})
  if not(gt!=255).any():missing.append(dict(position=r['position_id'],period=r['period'],reason='All-255 empty template is not GT'));continue
  validated.append((r,p,gt))
 if missing:
  status=dict(status='NOT_EVALUATED_NO_HUMAN_GT',Precision=None,Recall=None,locked_input_frames=len(rows),new_class_scored_frames=len(scored),
   baseline_frames_excluded=len(rows)-len(scored),missing_human_GT=missing)
  (a.output/'New_root_evaluation_status.json').write_text(json.dumps(status,indent=2));print('NOT_EVALUATED_NO_HUMAN_GT',len(missing));raise SystemExit(3)
 perframe=[];tubes={};gt_hashes={}
 for r,p,gt in validated:
  suffix='_new_mask.png'if a.prediction_encoding=='new-mask'else'_classes.png';pred=np.array(Image.open(a.prediction_dir/r['position_id']/(r['period']+suffix)))
  assert pred.shape==gt.shape and pred.ndim==2
  assert set(np.unique(pred)).issubset({0,255}if a.prediction_encoding=='new-mask'else{0,1,2,3})
  y=gt==(2 if a.gt_encoding=='temporal' else 1);z=pred==(255 if a.prediction_encoding=='new-mask' else 2);v=gt!=255
  tp=int((v&y&z).sum());fp=int((v&~y&z).sum());fn=int((v&y&~z).sum())
  perframe.append(dict(position=r['position_id'],period=r['period'],human_valid_pixels=int(v.sum()),**stats(tp,fp,fn)))
  d=tubes.setdefault(r['tube_id'],[0,0,0]);d[0]+=tp;d[1]+=fp;d[2]+=fn;gt_hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 tmetrics={t:stats(*x)for t,x in tubes.items()};pooled=stats(*[sum(x[i]for x in tubes.values())for i in range(3)])
 result=dict(status='EVALUATED',scope='Locked cross-tube human-GT evaluation; newly observed roots only',positive_class='Newly observed root',
  temporal_reference='Previous actually observed date',other_valid_pixels_remain_negatives=True,baseline_excluded=True,
  ground_truth_encoding=a.gt_encoding,prediction_encoding=a.prediction_encoding,per_frame=perframe,per_tube=tmetrics,pooled=pooled,
  tube_mean={k:float(np.mean([d[k]for d in tmetrics.values()if d[k]is not None]))if any(d[k]is not None for d in tmetrics.values())else None for k in ['Precision','Recall']},GT_SHA256=gt_hashes)
 (a.output/'New_root_evaluation_results.json').write_text(json.dumps(result,indent=2));print('EVALUATED_NEW_ONLY',len(tubes),'tubes',len(scored),'non-baseline frames')
if __name__=='__main__':main()
