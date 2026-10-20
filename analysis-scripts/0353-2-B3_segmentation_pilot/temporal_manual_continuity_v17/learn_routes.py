"""Fit junction affinities on training-window manual semantics only.

The raster labels have no root-instance IDs: same-class straight pairs are
weak positives, not independent biological identity ground truth.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import minimize
from scipy.special import expit
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--model',choices=c.MODELS,required=True);a=p.parse_args();r=a.run;name=a.model;out=r/'routing'/name;out.mkdir(parents=True,exist_ok=True);split=np.zeros((1108,2541),np.uint8)
for row in c.rows(r/'data/spatial_manifest.csv'):
 if row['split']=='train':
  x,y=int(row['x']),int(row['y']);split[y:y+256,x:x+256]=1
previous=None;samples=[];history=[]
for t in c.PERIODS:
 rgb=np.array(Image.open(r/'data/upper'/f'{t}_RGB.png'));gate=np.array(Image.open(a.roots/name/'upper'/f'{t}_root_mask.png'))>0;collected=[];axis=c.axes.build(gate,rgb,previous,collect=collected);previous=axis
 if t=='T1':continue
 gt=np.array(Image.open(r/'data/original'/f'{t}_classes.png'));known=(gt>0)&(gt<255);dist,near=ndi.distance_transform_edt(~known,return_indices=True);nearest=gt[tuple(near)]
 for n,item in enumerate(collected):
  arm_labels=[];purities=[];valid=True
  for key in ['arm_a','arm_b']:
   xy=np.array(item[key],int)
   if len(xy)<8:valid=False;break
   yy,xx=xy.T;good=(dist[yy,xx]<=3)&(split[yy,xx]>0)
   # At least half the distal arm must be manually typed within training area.
   if good.sum()<8 or good.mean()<.50:valid=False;break
   vals=nearest[yy[good],xx[good]];counts=np.bincount(vals,minlength=4);counts[0]=0;lab=int(counts.argmax());purity=counts[lab]/len(vals)
   if lab not in [1,2,3] or purity<.85:valid=False;break
   arm_labels.append(lab);purities.append(float(purity))
  if not valid:continue
  la,lb=arm_labels;f=np.array(item['features']);target=None;weight=None;basis=None
  if 2 in arm_labels and la!=lb:target=0.;weight=1.;basis='manual newly-observed versus persistent/growth: separate roots'
  elif la==lb and max(f[:3])<-.80:target=1.;weight=.5;basis='same manual class and straight multi-scale geometry: weak continuity positive'
  elif la==lb and min(f[:3])>-.1:target=0.;weight=.25;basis='same class, orthogonal arms: weak geometric negative'
  if target is None:continue
  samples.append(dict(period=t,junction_point=item['point'],features=item['features'],target=target,weight=weight,manual_arm_classes=arm_labels,purities=purities,basis=basis,training_pixels_only=True))
 print('ROUTER_SAMPLES',name,t,len(samples),flush=True)
assert len(samples)>=8,'Insufficient training junction evidence';X=np.array([v['features'] for v in samples]);y=np.array([v['target'] for v in samples]);w=np.array([v['weight'] for v in samples]);assert len(np.unique(y))==2,'Need both continue and separate examples';mean=X.mean(0);scale=np.maximum(X.std(0),.05);z=(X-mean)/scale
# Balanced data term, regularized small linear affinity; no test/val selection.
for label in [0,1]:w[y==label]/=w[y==label].sum()
w/=w.sum()
def objective(beta):
 score=z@beta[:-1]+beta[-1];loss=float(np.sum(w*(np.logaddexp(0,score)-y*score))+.05*np.sum(beta[:-1]**2));res=w*(expit(score)-y);grad=np.r_[z.T@res+.1*beta[:-1],res.sum()];history.append(loss);return loss,grad
fit=minimize(objective,np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-10});pr=expit(z@fit.x[:-1]+fit.x[-1]);meta=dict(version='v17_manual_semantic_junction_router',model=name,features=['dot12','dot24','dot40','RGB_chroma_distance','RGB_luminance_distance','log_width_ratio','same_previous_axis','min_previous_support','max_previous_support','previous_support_difference','min_arm_length_over100','multi_scale_direction_stability'],mean=mean.tolist(),scale=scale.tolist(),coef=fit.x[:-1].tolist(),intercept=float(fit.x[-1]),samples=len(y),positives=int(y.sum()),negatives=int((y==0).sum()),optimizer='L-BFGS-B, balanced weighted logistic loss, L2=.05',optimizer_success=bool(fit.success),iterations=int(fit.nit),final_objective=float(fit.fun),training_fit_accuracy=float(((pr>=.5)==y).mean()),evaluation_warning='Fit diagnostic only. Same-class straight positives are weak labels, not root-instance identity truth. No validation/test pixels used.',label_source='Exact original human temporal classes in training windows only',independent_root_identity_labels=False,original_human_SHA256=c.sha(r/'data/original_human_progress.json'))
c.js(out/'router.json',meta);c.js(out/'training_samples.json',samples);c.js(out/'optimization_history.json',history);print('ROUTER_FITTED',name,len(y),int(y.sum()),fit.success,flush=True)
