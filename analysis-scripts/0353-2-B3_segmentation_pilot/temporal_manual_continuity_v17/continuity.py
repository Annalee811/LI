"""Learned-emission root grammar with soft previous-date evidence.

Each putative axis is NEW, or terminal GROWTH / PERSISTENT / terminal GROWTH.
Unlike the strict geometric decoder, a missed previous binary tip does not erase
all learned growth evidence. Such growth is flagged for review, not verified.
"""
import numpy as np
from scipy.spatial import cKDTree
from structured import decode as strict_decode

def decode(probability,previous,current,link,strength=0.,mode='learned'):
 if mode=='strict':return strict_decode(probability,previous,current,link,strength)
 xy=current['xy'];mask=current['axis_map']>0
 if '_width_owner' not in current:
  owner=np.zeros(mask.shape,np.int32);offset=0
  for ai,path in enumerate(current['paths'],1):
   yy,xx=np.where(current['axis_map']==ai)
   if len(yy):_,ii=cKDTree(xy[path]).query(np.stack([yy,xx],1));owner[yy,xx]=offset+ii
   offset+=len(path)
  current['_width_owner']=owner
 owner=current['_width_owner'];size=sum(len(p) for p in current['paths']);ids=owner[mask];counts=np.maximum(np.bincount(ids,minlength=size),1);emission=np.stack([np.bincount(ids,weights=np.log(np.maximum(q[mask],1e-7)),minlength=size)/counts for q in probability],1)
 labels=[];records=[];offset=0
 for ai,path in enumerate(current['paths'],1):
  q=emission[offset:offset+len(path)].copy();offset+=len(path);n=len(path);info=link['axis_info'][ai-1];score=link['node_score'][path]
  q[:,0]+=strength*.4*(2*score-1);q[:,1]+=strength*.4*(1-2*score)
  # Three monotone states: growth-prefix, persistent-core, growth-suffix.
  dp=np.array([q[0,2],q[0,0],-np.inf]);back=np.zeros((n,3),np.uint8);penalty=.5
  for i in range(1,n):
   candidates=np.array([[dp[0],-np.inf,-np.inf],[dp[0]-penalty,dp[1],-np.inf],[dp[0]-penalty,dp[1]-penalty,dp[2]]]);back[i]=candidates.argmax(1);dp=candidates.max(1)+q[i,[2,0,2]]
  state=int(dp.argmax());pg_score=float(dp[state]);chosen=np.empty(n,np.uint8)
  for i in range(n-1,-1,-1):chosen[i]=1 if state==1 else 3;state=int(back[i,state]) if i else state
  new_score=float(q[:,1].sum())
  if new_score>=pg_score:chosen[:]=2;mode_name='newly_observed_axis'
  else:mode_name='learned_persistent_axis_with_terminal_growth' if np.any(chosen==1) else 'learned_growth_fragment_requires_tip_review'
  growth=int((chosen==3).sum());labels.append(chosen);records.append(dict(**info,mode=mode_name,new_nodes=int((chosen==2).sum()),persistent_nodes=int((chosen==1).sum()),growth_nodes=growth,new_and_growth_on_same_axis=False,decision_margin=abs(pg_score-new_score)/max(n,1),growth_continuation_previous_axis=info['matched_previous_axis'] if growth else 0,growth_biological_identity_verified=False))
 prediction=np.concatenate(labels)[owner];prediction[~mask]=0;assert np.array_equal(prediction>0,mask);assert all(not(v['new_nodes'] and v['growth_nodes']) for v in records);return prediction,records
