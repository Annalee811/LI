"""Root birth state and terminal-only extension on geometry-first joined roots."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from continuity import decode as loose_decode

def extension_evidence(previous,current,path):
 pts=current['xy'][path];n=len(path)
 if n<12:return False
 if '_extension_tips' not in previous:
  points=[];directions=[]
  for ppath,kinds in zip(previous['paths'],previous['endkinds']):
   if len(ppath)<20:continue
   pp=previous['xy'][ppath]
   for k,kind in [(0,kinds[0]),(-1,kinds[1])]:
    if kind!='tip':continue
    d=pp[0]-pp[min(8,len(pp)-1)] if k==0 else pp[-1]-pp[max(0,len(pp)-9)];points.append(pp[k]);directions.append(d/max(np.linalg.norm(d),1e-6))
  previous['_extension_tips']=(np.array(points).reshape(-1,2),np.array(directions).reshape(-1,2))
 anchors,directions=previous['_extension_tips']
 for endpoint,direction in [(pts[0],pts[min(10,n-1)]-pts[0]),(pts[-1],pts[max(0,n-11)]-pts[-1])]:
  direction=direction/max(np.linalg.norm(direction),1e-6)
  if np.any((np.linalg.norm(anchors-endpoint,axis=1)<=8)&((directions@direction)>.90)):return True
 return False

def decode(probability,previous,current,link,strength=1.,mode='identity'):
 if mode=='loose':return loose_decode(probability,previous,current,link,strength,'learned')
 xy=current['xy'];mask=current['axis_map']>0
 if '_width_owner' not in current:
  owner=np.zeros(mask.shape,np.int32);offset=0
  for ai,path in enumerate(current['paths'],1):
   yy,xx=np.where(current['axis_map']==ai)
   if len(yy):_,nn=cKDTree(xy[path]).query(np.stack([yy,xx],1));owner[yy,xx]=offset+nn
   offset+=len(path)
  current['_width_owner']=owner
 owner=current['_width_owner'];size=sum(len(p) for p in current['paths']);ids=owner[mask];counts=np.maximum(np.bincount(ids,minlength=size),1);emission=np.stack([np.bincount(ids,weights=np.log(np.maximum(v[mask],1e-7)),minlength=size)/counts for v in probability],1);labels=[];records=[];offset=0
 for ai,path in enumerate(current['paths'],1):
  n=len(path);q=emission[offset:offset+n].copy();offset+=n;info=link['axis_info'][ai-1];score=link['node_score'][path];sm=ndi.median_filter(score,size=min(7,2*(n//2)+1),mode='nearest');good=sm>.4;support=float(good.mean());old_ids=np.flatnonzero(good);has_core=len(old_ids)>=max(8,int(.12*n));extension=extension_evidence(previous,current,path) if not has_core else False
  # Previous visibility applies to the root as a whole, not only one fragment.
  novelty=max(0.,(.35-support)/.35);q[:,1]+=strength*1.5*novelty;q[:,0]+=strength*.7*sm;q[:,2]-=strength*2.0*sm
  allowed=np.zeros(n,bool)
  if has_core:
   first,last=int(old_ids[0]),int(old_ids[-1]);
   if first>=6:allowed[:first]=True
   if n-last-1>=6:allowed[last+1:]=True
  elif extension:allowed[:]=True
  q[~allowed,2]=-1e6
  # Monotone G-prefix / P-core / G-suffix; no growth island in an old shaft.
  dp=np.array([q[0,2],q[0,0],-np.inf]);back=np.zeros((n,3),np.uint8)
  for i in range(1,n):
   mat=np.array([[dp[0],-np.inf,-np.inf],[dp[0]-.5,dp[1],-np.inf],[dp[0]-.5,dp[1]-.5,dp[2]]]);back[i]=mat.argmax(1);dp=mat.max(1)+q[i,[2,0,2]]
  state=int(dp.argmax());pg=float(dp[state]);chosen=np.empty(n,np.uint8)
  for i in range(n-1,-1,-1):chosen[i]=1 if state==1 else 3;state=int(back[i,state]) if i else state
  novel=float(q[:,1].sum())
  if novel>=pg:chosen[:]=2;kind='newly_observed_whole_root'
  elif (chosen==3).any():kind='persistent_root_with_evidence_gated_terminal_growth' if has_core else 'previous_tip_growth_continuation'
  else:kind='persistent_whole_root'
  labels.append(chosen);records.append(dict(**info,mode=kind,new_nodes=int((chosen==2).sum()),persistent_nodes=int((chosen==1).sum()),growth_nodes=int((chosen==3).sum()),new_and_growth_on_same_axis=False,decision_margin=abs(pg-novel)/max(n,1),root_previous_visibility_fraction=support,growth_allowed_by_previous_core=has_core,previous_tip_continuation=extension,growth_continuation_previous_axis=info['matched_previous_axis'] if (chosen==3).any() else 0,growth_biological_identity_verified=False))
 pred=np.concatenate(labels)[owner];pred[~mask]=0;assert np.array_equal(pred>0,mask);return pred,records
