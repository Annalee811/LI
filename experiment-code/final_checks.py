from pathlib import Path
import json,csv,hashlib
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parents[1];p=json.loads((W/'reports/locked_protocol.json').read_text());rows=[]
for v in p['models']:
 for seed in p['seeds']:
  d=W/'experiments'/v/str(seed);r=json.loads((d/'result.json').read_text());h=list(csv.DictReader((d/'training_history.csv').open(encoding='utf-8-sig')));assert len(h)==100 and[int(q['epoch'])for q in h]==list(range(1,101));expected=max(h,key=lambda q:float(q['validation_macro_Dice']));assert r['selected_epoch']==int(expected['epoch']);assert r['test']['selection_used_test']is False
  z=np.load(d/'inference.npz');assert all(np.isfinite(z[k]).all()for k in z.files)
  for t in ['T1','T2','T3','T4']:
   a=np.array(Image.open(d/f'predictions/{t}_classes.png'));b=np.array(Image.open(W/f'data/baseline_prior/{t}_classes.png'));assert a.shape==b.shape and np.array_equal(a>0,b>0)
  rows.append(dict(variant=v,seed=seed,epochs=100,validation_checkpoint_selection_verified=True,finite_neural_outputs=True,best_checkpoint_SHA256=hashlib.sha256((d/'best.pt').read_bytes()).hexdigest()))
assert len(rows)==15
manifest=list(csv.DictReader((W/'reports/annotation_checklist.csv').open(encoding='utf-8-sig')));assert len(manifest)==26 and len(set(r['tube_id']for r in manifest))==3
for r in manifest:
 gt=np.array(Image.open(r['GT_path']));assert(gt==255).all();assert r['annotation_state']=='pending';assert r['tube_id']!='0353-2'
 for v in p['models']:
  for seed in p['seeds']:
   path=W/'independent_predictions'/v/str(seed)/r['position_id']/(r['period']+'_classes.png');a=np.array(Image.open(path));base=Path('/workspace/work/V90_all_microtubes_20261118/transfer_candidates')/r['position_id']/'base/unet/full'/(r['period']+'_classes.png');b=np.array(Image.open(base));assert a.shape==b.shape and np.array_equal(a>0,b>0)
assert json.loads((W/'reports/independent_test/evaluation_status.json').read_text())['metrics']is None
report=dict(passed=True,formal_runs=15,updates_per_run=100,total_training_updates=1500,independent_prediction_masks=390,independent_tubes=3,independent_frames=26,external_accuracy_evaluated=False,independent_status='NOT_EVALUATED_NO_HUMAN_GT',checks=rows);(W/'reports/verification.json').write_text(json.dumps(report,indent=2));print('VERIFIED_15_TRAINING_RUNS_AND_390_EXTERNAL_PREDICTIONS')
