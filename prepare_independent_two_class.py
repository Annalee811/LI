"""Group the existing locked cross-tube predictions; do not manufacture GT."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
from PIL import Image
W=Path(__file__).resolve().parent;S=Path('/workspace/work/V90_validation_20261123')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 with(W/'Independent_manifest.csv').open(encoding='utf-8-sig')as f:rows=list(csv.DictReader(f))
 assert len(rows)==26 and len(set(r['tube_id']for r in rows))==3
 dest=W/'independent_two_class_predictions';records=[]
 for p in sorted((S/'independent_predictions').rglob('*_classes.png')):
  rel=p.relative_to(S/'independent_predictions');out=dest/rel.parent/(p.stem.replace('_classes','_two_classes')+'.png');out.parent.mkdir(parents=True,exist_ok=True)
  cl=np.array(Image.open(p));mapped=np.zeros(cl.shape,np.uint8);mapped[np.isin(cl,[1,3])]=1;mapped[cl==2]=2;mapped[np.isin(cl,[4,255])]=255
  assert np.array_equal(mapped==2,cl==2)and np.array_equal(mapped>0,cl>0)
  Image.fromarray(mapped).save(out)
  records.append(dict(source=str(rel),output=str(out.relative_to(W)),source_SHA256=sha(p),output_SHA256=sha(out),GT_available=False,metrics_status='NOT_EVALUATED_NO_HUMAN_GT'))
 assert len(records)==442
 with(W/'Independent_prediction_index.csv').open('w',encoding='utf-8-sig',newline='')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 # A single pre-fixed optimization seed is used for examples, never chosen by test performance.
 examples=[];cache=W/'figure3_image_cache';cache.mkdir(exist_ok=True)
 for tube in ['0353-3','A017-3','Q1-2']:
  pos=sorted(set(r['position_id']for r in rows if r['tube_id']==tube))[0];rr=sorted([r for r in rows if r['position_id']==pos],key=lambda r:r['period'])
  assert len(rr)==4
  prior=S/'independent_predictions/V66_pairwise_frozen/frozen'/pos
  masks=[np.array(Image.open(prior/(r['period']+'_classes.png')))>0 for r in rr[1:]]
  height,width=masks[0].shape;size=256
  ints=[np.pad(m.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)for m in masks];best=None
  for y in range(0,height-size+1,16):
   for x in range(0,width-size+1,16):
    counts=[int(z[y+size,x+size]-z[y,x+size]-z[y+size,x]+z[y,x])for z in ints];score=sum(np.log1p(c)for c in counts)
    if best is None or score>best[0]:best=(score,x,y,counts)
  _,x,y,counts=best;crop=np.s_[y:y+size,x:x+size];data={}
  for r in rr:
   t=r['period'];rgbpath=Path(r['RGB_path']);assert sha(rgbpath)==r['RGB_SHA256']
   rgb=np.array(Image.open(rgbpath).convert('RGB'))[crop]
   full=np.array(Image.open(dest/'full/20261123'/pos/(t+'_two_classes.png')))[crop]
   data[t+'_RGB']=rgb;data[t+'_prediction']=full
  np.savez_compressed(cache/(pos+'.npz'),**data)
  examples.append(dict(tube=tube,position=pos,x=x,y=y,width=size,height=size,periods=[r['period']for r in rr],
   crop_selection='Max sum log(1+common frozen-foreground pixels) over T2–T4, grid16; no GT, confidence or model-agreement selection',
   checkpoint_variant='full',seed=20261123,common_foreground_counts=counts))
 (W/'Independent_examples.json').write_text(json.dumps(examples,indent=2))
 pending=dict(status='NOT_EVALUATED_NO_HUMAN_GT',test_tubes=['0353-3','A017-3','Q1-2'],positions=7,frames=26,baseline_frames=7,temporal_frames=19,
  classes=['existing','new'],Precision=None,Recall=None,Dice=None,IoU=None,macro_Dice=None,GT_annotation_state='pending',
  training_tube_0353_2_excluded=True,prior_unlabelled_image_exposure=True,prepared_feature_adapter_has_documented_approximations=True)
 (W/'Independent_evaluation_status.json').write_text(json.dumps(pending,indent=2))
 print('INDEPENDENT_TWO_CLASS_PREDICTIONS_READY',len(records),'masks; actual independent metrics remain undefined',flush=True)
if __name__=='__main__':main()
