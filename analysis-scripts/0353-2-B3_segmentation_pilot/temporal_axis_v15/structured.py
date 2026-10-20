"""Active whole-axis decoder with unique width ownership at crossings."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from axes import tangents

def _emission(probability,axis):
 xy=axis['xy'];log=np.log(np.maximum(probability,1e-7));return log[:,xy[:,0],xy[:,1]].T

def decode(probability,previous,current,link,strength=1.):
 """Each axis: entirely NEW, or terminal GROWTH + PERSISTENT + terminal GROWTH.

Growth must connect to an interior previous tip with compatible tangent and
unmatched terminal evidence. NEW and GROWTH cannot coexist on one accepted axis.
"""
 xy=current['xy'];emission=_emission(probability,current);labels=np.zeros(len(xy),np.uint8);records=[];axis_labels=[];h,w=current['axis_map'].shape
 tip_points=[];tip_outward=[];tip_axis_ids=[]
 for pi,ppath in enumerate(previous['paths'],1):
  if len(ppath)<12:continue
  ppoints=previous['xy'][ppath];tt=tangents(ppoints)
  for end,sign,kind in [(0,-1,previous['endkinds'][pi-1][0]),(-1,1,previous['endkinds'][pi-1][1])]:
   if kind=='tip':tip_points.append(ppoints[end]);tip_outward.append(sign*tt[end]);tip_axis_ids.append(pi)
 tip_tree=cKDTree(np.array(tip_points)) if tip_points else None
 for ai,path in enumerate(current['paths'],1):
  q=emission[path].copy();info=link['axis_info'][ai-1];score=link['node_score'][path];n=len(path);old=info['matched_previous_axis'];support=info['previous_support_fraction'];dominance=info['dominant_previous_fraction'];pts=xy[path];allowed=[]
  # Width-normalized averages prevent a long/thick axis from dominating others.
  # Geometry is an interpretable prior, not a substitute for RGB appearance.
  q[:,0]+=strength*1.2*(2*score-1);q[:,1]+=strength*1.2*(1-2*score);q[:,2]+=strength*(.6-2*score)
  new_score=float(q[:,1].mean());old_score=-np.inf;best_labels=np.ones(n,np.uint8);prefix=[0];suffix=[n]
  if old and support>=.10 and dominance>=.45:
   pp=previous['xy'][previous['paths'][old-1]];pkinds=previous['endkinds'][old-1];ptan=tangents(pp);ctan=tangents(pts)
   for tip_index,kind in [(0,pkinds[0]),(-1,pkinds[1])]:
    if kind!='tip':continue
    dist=np.linalg.norm(pts-pp[tip_index],axis=1);candidate=np.flatnonzero((dist<=16)&(np.abs(ctan@ptan[tip_index])>=.60))
    if not len(candidate):continue
    center=int(candidate[np.argmin(dist[candidate])]);
    for boundary in sorted(set(int(np.clip(center+d,1,n-1)) for d in [-6,-3,0,3,6])) if n>1 else []:
     if boundary>=3 and boundary<n-5 and current['endkinds'][ai-1][0] in ['tip','junction','unresolved'] and score[:boundary].mean()<.35 and float((score[boundary:]>.4).mean())>.15:prefix.append(boundary)
     if n-boundary>=3 and boundary>5 and current['endkinds'][ai-1][1] in ['tip','junction','unresolved'] and score[boundary:].mean()<.35 and float((score[:boundary]>.4).mean())>.15:suffix.append(boundary)
   cumP=np.r_[0,np.cumsum(q[:,0])];cumG=np.r_[0,np.cumsum(q[:,2])]
   for lo in prefix:
    for hi in suffix:
     if hi-lo<max(3,int(.1*n)):continue
     value=float((cumG[lo]+cumP[hi]-cumP[lo]+cumG[n]-cumG[hi])/n)-.03*((lo>0)+(hi<n))
     if value>old_score:old_score=value;best_labels=np.ones(n,np.uint8);best_labels[:lo]=3;best_labels[hi:]=3
   # Good distributed previous overlap rules out a NEW interior on this axis.
   must_old=support>=.70 and dominance>=.80 and old_score>=new_score-.2
  else:must_old=False
  if np.isfinite(old_score) and (must_old or old_score>=new_score):chosen=best_labels;mode='persistent_axis_with_optional_tip_growth';margin=old_score-new_score
  else:chosen=np.full(n,2,np.uint8);mode='newly_observed_axis';margin=new_score-old_score if np.isfinite(old_score) else None
  # A resolved growth continuation may be separated from its old shaft by an
  # unresolved crossing. Absence of an old segment inside this trace does not
  # make it a new lateral. Require outward connection to a real previous tip,
  # little previous overlap, and learned GROWTH evidence over the alternatives.
  extension_parent=0
  if tip_tree is not None and n>=8 and support<.20:
   tangent=tangents(pts)
   for endpoint,direction in [(pts[0],tangent[0]),(pts[-1],-tangent[-1])]:
    dd,ii=tip_tree.query(endpoint,k=min(4,len(tip_points)),distance_upper_bound=12)
    for distance,tip in zip(np.atleast_1d(dd),np.atleast_1d(ii)):
     if tip<len(tip_points) and float(direction@tip_outward[tip])>=.70:extension_parent=tip_axis_ids[tip];break
    if extension_parent:break
   growth_score=float(q[:,2].mean())-.08
   if extension_parent and growth_score>max(new_score,old_score):chosen=np.full(n,3,np.uint8);mode='tip_growth_continuation_at_crossing';margin=growth_score-max(new_score,old_score)
  labels[path]=chosen;axis_labels.append(chosen.copy());records.append(dict(**info,mode=mode,growth_continuation_previous_axis=extension_parent if mode=='tip_growth_continuation_at_crossing' else 0,growth_nodes=int((chosen==3).sum()),persistent_nodes=int((chosen==1).sum()),new_nodes=int((chosen==2).sum()),new_and_growth_on_same_axis=False,prefix_growth_nodes=int(np.flatnonzero(chosen!=3)[0]) if np.any(chosen!=3) else n,suffix_growth_nodes=int(n-1-np.flatnonzero(chosen!=3)[-1]) if np.any(chosen!=3) else 0,decision_margin=float(margin) if margin is not None else None))
 seed=np.zeros(current['axis_map'].shape,bool);seed[tuple(xy.T)]=True;_,near=ndi.distance_transform_edt(~seed,return_indices=True);index=np.zeros(seed.shape,np.int32);index[tuple(xy.T)]=np.arange(len(xy));prediction=labels[index[tuple(near)]];mask=current['axis_map']>0;prediction[~mask]=0
 # Where a nearby different axis owns the width, use its nearest own skeleton.
 for ai,path in enumerate(current['paths'],1):
  region=current['axis_map']==ai
  if not region.any():continue
  ry,rx=np.where(region);_,nn=cKDTree(xy[path]).query(np.stack([ry,rx],1));prediction[ry,rx]=axis_labels[ai-1][nn]
 assert np.array_equal(prediction>0,mask)
 assert all(not (v['new_nodes'] and v['growth_nodes']) for v in records)
 return prediction,records
