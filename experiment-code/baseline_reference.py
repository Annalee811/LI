from pathlib import Path
import json,numpy as np
from PIL import Image
from train_study import metrics
W=Path(__file__).resolve().parents[1];O=Path('/workspace/work/V90_all_microtubes_20261118');sources={'V66_pairwise_frozen':O/'model_source/assets/base/unet/full','V90_automatic_historical_reference':O/'B3_verified_inference/junction/full'};stages={'local_input':O/'B3_verified_inference/local/full','junction_pair_decoder':O/'B3_verified_inference/joint/final/full','branch_consistency':O/'B3_verified_inference/branch/full','axis_consistency':O/'B3_verified_inference/axis/full','junction_consistency':O/'B3_verified_inference/junction/full'};results={}
for name,folder in {**sources,**stages}.items():
 row={}
 for split in [0,1,2]:
  cm=np.zeros((3,3),np.int64)
  for t in ['T2','T3','T4']:
   q=np.load(W/f'data/{t}_pixels.npz');ok=q['split']==split;pred=np.array(Image.open(folder/(t+'_classes.png'))).ravel()[q['pixel'][ok]]-1;assert np.isin(pred,[0,1,2]).all();cm+=np.bincount(q['target'][ok]*3+pred,minlength=9).reshape(3,3)
  row[str(split)]=metrics(cm)
 results[name]=dict(metrics_by_split=row,scope='Existing frozen automatic outputs re-evaluated on the same eligible pixels; historical B3 exposure. Not fresh retraining and not independent testing.')
# Exact log-probability prior serves as the zero-residual baseline of the controlled heads.
row={}
for split in [0,1,2]:
 cm=np.zeros((3,3),np.int64)
 for t in ['T2','T3','T4']:
  q=np.load(W/f'data/{t}_pixels.npz');ok=q['split']==split;pred=q['pf'][ok,:3].argmax(1);cm+=np.bincount(q['target'][ok]*3+pred,minlength=9).reshape(3,3)
 row[str(split)]=metrics(cm)
results['pairwise_prior_zero_residual']=dict(metrics_by_split=row,scope='Frozen released prior probabilities; common zero-residual starting point for temporal variants')
(W/'reports/frozen_references.json').write_text(json.dumps(results,indent=2));print({k:round(v['metrics_by_split']['2']['macro_Dice'],6)for k,v in results.items()})
