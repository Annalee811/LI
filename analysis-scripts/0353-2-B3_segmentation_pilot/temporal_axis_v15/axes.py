"""RGB-supported root-axis tracing and whole-axis temporal grammar.

No manual labels are read here. Axes are putative trajectories, not established
biological identities. Ambiguous intersections are flagged, never silently fused.
"""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

def root_contrast(rgb):
 lum=rgb.astype('float32').mean(2)/255
 # Lighting-invariant local foreground contrast; raw RGB values remain unchanged.
 return np.clip((lum-ndi.gaussian_filter(lum,5))/.12,-1,1).astype('float32')

def _graph(mask):
 sk=skeletonize(mask);xy=np.argwhere(sk);lookup={tuple(p):i for i,p in enumerate(xy)};adj=[[] for _ in xy]
 for i,(y,x) in enumerate(xy):
  for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
   j=lookup.get((y+dy,x+dx))
   if j is not None and not(dy and dx and ((y+dy,x) in lookup or (y,x+dx) in lookup)):adj[i].append(j)
 return sk,xy,adj

def _matchings(indices):
 if len(indices)<2:return [()]
 first=indices[0];res=[v for v in _matchings(indices[1:])]
 for j in indices[1:]:
  for rest in _matchings([v for v in indices[1:] if v!=j]):res.append(((first,j),)+rest)
 return res

def build(mask,rgb,previous=None):
 sk,xy,adj=_graph(mask);h,w=mask.shape;n=len(xy);degree=np.array([len(v) for v in adj]);junction=np.zeros(mask.shape,bool)
 if n:junction[tuple(xy[degree>=3].T)]=True
 clusters,num=ndi.label(ndi.binary_dilation(junction,iterations=2),structure=np.ones((3,3)))
 inside=clusters[tuple(xy.T)] if n else np.array([],int);vertices={};vid=np.full(n,-1,np.int32)
 for k in np.unique(inside[inside>0]):
  ii=np.flatnonzero(inside==k);v=len(vertices);vertices[v]=dict(nodes=ii.tolist(),kind='junction',point=xy[ii].mean(0).tolist());vid[ii]=v
 for i in np.flatnonzero((degree!=2)&(inside==0)):
  v=len(vertices);y,x=xy[i];vertices[v]=dict(nodes=[int(i)],kind='edge' if min(y,x,h-1-y,w-1-x)<4 else 'tip',point=xy[i].tolist());vid[i]=v
 # Components with no endpoint are loops, not silently deleted.
 seen=set();edges=[]
 for start in range(n):
  if vid[start]<0:continue
  for nxt in adj[start]:
   key=tuple(sorted((start,nxt)))
   if key in seen or (vid[nxt]==vid[start]):seen.add(key);continue
   path=[start,nxt];seen.add(key);prev,cur=start,nxt
   while vid[cur]<0:
    choices=[j for j in adj[cur] if j!=prev]
    if not choices:break
    nxt=choices[0];key=tuple(sorted((cur,nxt)))
    if key in seen:break
    seen.add(key);path.append(nxt);prev,cur=cur,nxt
   if vid[cur]<0:
    v=len(vertices);vertices[v]=dict(nodes=[int(cur)],kind='unresolved',point=xy[cur].tolist());vid[cur]=v
   if len(path)>=3:edges.append(dict(nodes=path,ends=[int(vid[start]),int(vid[cur])]))
 for i in range(n):
  for j in adj[i]:
   key=tuple(sorted((i,j)))
   if key in seen or vid[i]>=0 or vid[j]>=0:continue
   path=[i,j];seen.add(key);prev,cur=i,j
   for _ in range(n):
    choices=[v for v in adj[cur] if v!=prev and tuple(sorted((cur,v))) not in seen]
    if not choices:break
    nxt=choices[0];seen.add(tuple(sorted((cur,nxt))));path.append(nxt);prev,cur=cur,nxt
    if cur==i:break
   if len(path)>=3:
    v=len(vertices);vertices[v]=dict(nodes=[i],kind='loop',point=xy[i].tolist());edges.append(dict(nodes=path,ends=[v,v]))
 arms={k:[] for k in vertices};width=ndi.distance_transform_edt(mask);color=rgb.astype('float32')/255
 for ei,e in enumerate(edges):
  for side,v in enumerate(e['ends']):
   nodes=e['nodes'] if side==0 else e['nodes'][::-1];pts=xy[nodes];q=pts[min(12,len(pts)-1)]-pts[0];q=q/max(float(np.linalg.norm(q)),1e-6);idx=pts[1:min(len(pts),16)];appearance=np.median(color[tuple(idx.T)],0);appearance-=appearance.mean();wid=float(np.median(width[tuple(idx.T)]));arms[v].append((ei,side,q,appearance,wid))
 # Previous-frame geometry can disambiguate an otherwise similar crossing.
 # This consumes predictions only, never previous human traces.
 prior={};previous_tree=None
 if previous is not None and len(previous['xy']):
  previous_tree=cKDTree(previous['xy']);previous_tangent=np.zeros((len(previous['xy']),2),np.float32)
  for path in previous['paths']:previous_tangent[path]=tangents(previous['xy'][path])
  previous_ids=previous['axis_map'][tuple(previous['xy'].T)]
 for aa in arms.values():
  for ei,side,direction,appearance,wid in aa:
   prior[ei,side]=(0,0.)
   if previous_tree is None:continue
   nn=edges[ei]['nodes'] if side==0 else edges[ei]['nodes'][::-1];points=xy[nn[3:min(40,len(nn)):3]]
   if not len(points):continue
   kk=min(4,len(previous['xy']));dd,ii=previous_tree.query(points,k=kk,distance_upper_bound=16)
   if kk==1:dd,ii=dd[:,None],ii[:,None]
   valid=ii<len(previous['xy']);safe=np.minimum(ii,len(previous['xy'])-1);al=np.abs((previous_tangent[safe]*direction).sum(2));sc=np.exp(-.5*(dd/7)**2)*al**4;sc[~valid]=0;best=sc.argmax(1);rr=np.arange(len(points));good=sc[rr,best]>.4;ids=previous_ids[safe[rr,best]];votes=np.bincount(ids[good],minlength=len(previous['paths'])+1);votes[0]=0;winner=int(votes.argmax())
   if winner:prior[ei,side]=(winner,float(votes[winner]/len(points)))
 links={};decisions=[];uncertain=np.zeros(mask.shape,bool)
 for v,aa in arms.items():
  if len(aa)<2:continue
  costs={}
  for i in range(len(aa)):
   for j in range(i+1,len(aa)):
    dot=float(aa[i][2]@aa[j][2]);cost=(1+dot)/2+.10*min(1,float(np.linalg.norm(aa[i][3]-aa[j][3]))/.20)+.06*min(1,abs(np.log((aa[i][4]+.5)/(aa[j][4]+.5))))
    pi,pj=prior[aa[i][:2]],prior[aa[j][:2]]
    if pi[0] and pj[0] and min(pi[1],pj[1])>=.5:cost+=(-.12 if pi[0]==pj[0] else .12)
    if dot<-.45 and cost<.34:costs[(i,j)]=cost
  # Maximal admissible pairings, with score/margin check. T junction: pair the
  # smooth main axis; the unpaired arm remains its own lateral-root axis.
  candidates=[]
  for pairs in _matchings(list(range(len(aa)))) if len(aa)<=8 else [()]:
   if pairs and all(tuple(sorted(pair)) in costs for pair in pairs):candidates.append((len(pairs),sum(costs[tuple(sorted(z))] for z in pairs),pairs))
  candidates.sort(key=lambda x:(-x[0],x[1]));chosen=();margin=None
  if candidates:
   best=candidates[0];others=[q for q in candidates[1:] if q[0]==best[0]];margin=others[0][1]-best[1] if others else 1.;supported=all(prior[aa[i][:2]][0] and prior[aa[i][:2]][0]==prior[aa[j][:2]][0] and min(prior[aa[i][:2]][1],prior[aa[j][:2]][1])>=.5 for i,j in best[2]);chosen=best[2] if margin>=(.02 if supported else .06) else ()
  elif len(aa)>8:
   # Dense junction clusters: accept only mutually best, well-separated pairs.
   available=set(range(len(aa)));accepted=[]
   for (i,j),value in sorted(costs.items(),key=lambda item:item[1]):
    if i not in available or j not in available:continue
    other_i=[z for (u,vv),z in costs.items() if i in (u,vv) and j not in (u,vv) and u in available and vv in available];other_j=[z for (u,vv),z in costs.items() if j in (u,vv) and i not in (u,vv) and u in available and vv in available]
    if min(other_i+[1.])-value>=.08 and min(other_j+[1.])-value>=.08:accepted.append((i,j));available-={i,j}
   chosen=tuple(accepted);margin=None
  if len(aa)>2 and (not chosen or margin is not None and margin<.10):
   point=np.rint(vertices[v]['point']).astype(int);uncertain[max(0,point[0]-5):point[0]+6,max(0,point[1]-5):point[1]+6]=True
  for i,j in chosen:
   a,b=aa[i][:2],aa[j][:2];links[a]=b;links[b]=a
  decisions.append(dict(vertex=v,degree=len(aa),point=vertices[v]['point'],pairs=[[int(aa[i][0]),int(aa[j][0])] for i,j in chosen],margin=margin,unresolved=bool(len(aa)>2 and not chosen),basis='RGB appearance, width, opposite tangent and previous-frame predicted-axis support'))
 # Short gaps may link two *existing* tips; no RGB pixels or root mask are filled.
 tips=[]
 for v,aa in arms.items():
  if vertices[v]['kind']=='tip' and len(aa)==1:tips.append((v,aa[0]))
 gap_candidates=[]
 if tips:
  tree=cKDTree(np.array([vertices[v]['point'] for v,_ in tips]));contrast=root_contrast(rgb)
  for i,j in tree.query_pairs(10):
   va,a=tips[i];vb,b=tips[j];pa=np.array(vertices[va]['point']);pb=np.array(vertices[vb]['point']);delta=pb-pa;dist=np.linalg.norm(delta);unit=delta/max(dist,1e-6)
   if dist<2 or a[0]==b[0] or float(a[2]@(-unit))<.80 or float(b[2]@unit)<.80:continue
   line=np.rint(pa[None]+np.linspace(0,1,max(3,int(dist)))[...,None]*delta).astype(int);contrast_value=float(np.median(contrast[tuple(line.T)]));rgb_delta=float(np.linalg.norm(a[3]-b[3]))
   if contrast_value<.025 or rgb_delta>.12:continue
   gap_candidates.append((dist,va,vb,a[:2],b[:2]))
 used=set()
 for dist,va,vb,a,b in sorted(gap_candidates):
  if va in used or vb in used:continue
  links[a]=b;links[b]=a;used|={va,vb};decisions.append(dict(kind='RGB_supported_virtual_gap',distance=float(dist),vertices=[va,vb],root_pixels_filled=0))
 # Concatenate edge paths across accepted intersections, rather than classifying
 # the intervals between junctions independently.
 consumed=set();paths=[];endkinds=[];edge_axes={}
 starts=[(i,s) for i in range(len(edges)) for s in [0,1] if (i,s) not in links]+[(i,0) for i in range(len(edges))]
 for ei,side in starts:
  if ei in consumed:continue
  nodes=[];first_kind=vertices[edges[ei]['ends'][side]]['kind'];cur,ss=ei,side;last_kind='loop'
  for _ in range(len(edges)+1):
   if cur in consumed:break
   consumed.add(cur);edge_axes[cur]=len(paths)+1;e=edges[cur];idx=e['nodes'] if ss==0 else e['nodes'][::-1];nodes+=idx;exit_side=1-ss;last_kind=vertices[e['ends'][exit_side]]['kind'];other=links.get((cur,exit_side))
   if other is None:break
   cur,ss=other
  if nodes:paths.append(np.array(nodes,dtype=np.int32));endkinds.append([first_kind,last_kind])
 axis_at=np.zeros(mask.shape,np.int32);owner_node=np.full(mask.shape,-1,np.int32)
 for ai,path in enumerate(paths,1):axis_at[tuple(xy[path].T)]=ai;owner_node[tuple(xy[path].T)]=path
 # Assign mask width only to a skeleton axis in its own binary component.
 seed=axis_at>0
 if seed.any():
  _,nearest=ndi.distance_transform_edt(~seed,return_indices=True);component,_=ndi.label(mask,structure=np.ones((3,3)));same=component==component[tuple(nearest)];take=mask&same;axis_at[take]=axis_at[tuple(nearest)][take]
 # Isolated junction blobs not in a trace are explicit unresolved singleton axes.
 left=mask&(axis_at==0);ll,nn=ndi.label(left,structure=np.ones((3,3)));node_index=np.full(mask.shape,-1,np.int32)
 if n:node_index[tuple(xy.T)]=np.arange(n)
 for k in range(1,nn+1):
  pts=np.argwhere((ll==k)&sk)
  if not len(pts):continue
  ids=node_index[tuple(pts.T)];paths.append(np.array(ids,np.int32));endkinds.append(['unresolved','unresolved']);axis_at[ll==k]=len(paths);uncertain[ll==k]=True
 # Ensure all residual mask pixels have a valid axis, without crossing gaps.
 remaining=mask&(axis_at==0)
 if remaining.any() and (axis_at>0).any():
  _,near=ndi.distance_transform_edt(axis_at==0,return_indices=True);axis_at[remaining]=axis_at[tuple(near)][remaining];uncertain[remaining]=True
 assert np.array_equal(axis_at>0,mask)
 return dict(xy=xy,paths=paths,endkinds=endkinds,axis_map=axis_at,uncertain=uncertain,decisions=decisions)

def tangents(points):
 n=len(points)
 if n==1:return np.zeros((1,2),np.float32)
 lo=np.maximum(np.arange(n)-6,0);hi=np.minimum(np.arange(n)+6,n-1);d=points[hi].astype(float)-points[lo];return (d/np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-6)).astype('float32')

def match(previous,current,previous_rgb,current_rgb):
 pxy=previous['xy'];cxy=current['xy'];ps=previous['axis_map'][tuple(pxy.T)];ctan=np.zeros((len(cxy),2),np.float32);ptan=np.zeros((len(pxy),2),np.float32)
 for path in previous['paths']:ptan[path]=tangents(pxy[path])
 for path in current['paths']:ctan[path]=tangents(cxy[path])
 nearest_id=np.zeros(len(cxy),np.int32);node_score=np.zeros(len(cxy),np.float32);nearest_distance=np.full(len(cxy),100,np.float32)
 if len(pxy) and len(cxy):
  k=min(8,len(pxy));dist,idx=cKDTree(pxy).query(cxy,k=k,distance_upper_bound=20)
  if k==1:dist,idx=dist[:,None],idx[:,None]
  valid=idx<len(pxy);safe=np.minimum(idx,len(pxy)-1);alignment=np.abs((ctan[:,None]*ptan[safe]).sum(2));score=np.exp(-.5*(dist/7)**2)*alignment**4;score[~valid]=0;best=score.argmax(1);ii=np.arange(len(cxy));node_score=score[ii,best].astype('float32');nearest_id=ps[safe[ii,best]];nearest_distance=dist[ii,best].astype('float32');nearest_distance[~np.isfinite(nearest_distance)]=100
 info=[];context=np.zeros((*current['axis_map'].shape,4),np.float32);node_context=np.zeros((len(cxy),4),np.float32)
 for ai,path in enumerate(current['paths'],1):
  score=node_score[path];ids=nearest_id[path];good=score>.4;counts=np.bincount(ids[good],minlength=len(previous['paths'])+1);counts[0]=0;dominant=int(counts.argmax()) if counts.max()>0 else 0;fraction=float(good.mean());dominant_fraction=float(counts[dominant]/max(1,good.sum())) if dominant else 0
  pts=cxy[path];steps=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))];total=max(steps[-1],1);arc=np.minimum(steps,total-steps)/total
  support=good.astype(bool);dist_to_support=ndi.distance_transform_edt(~support) if support.any() else np.full(len(path),128.);novel=np.minimum(dist_to_support,128)/128
  node_context[path]=np.stack([np.full(len(path),fraction),score,arc,novel],1)
  info.append(dict(axis_id=ai,nodes=len(path),length_native_pixels=float(total),matched_previous_axis=dominant,previous_support_fraction=fraction,dominant_previous_fraction=dominant_fraction,endpoint_kinds=current['endkinds'][ai-1],unresolved_crossing=bool(current['uncertain'][tuple(pts.T)].any())))
 seed=np.zeros(current['axis_map'].shape,bool);seed[tuple(cxy.T)]=True
 if seed.any():
  _,near=ndi.distance_transform_edt(~seed,return_indices=True);index=np.zeros(seed.shape,np.int32);index[tuple(cxy.T)]=np.arange(len(cxy));context=node_context[index[tuple(near)]];context[current['axis_map']==0]=0
 return dict(node_score=node_score,nearest_previous_axis=nearest_id,nearest_distance=nearest_distance,axis_info=info,context=context)

def _emission(probability,axis):
 xy=axis['xy'];log=np.log(np.maximum(probability,1e-7));return log[:,xy[:,0],xy[:,1]].T

def decode(probability,previous,current,link,strength=1.):
 """Each axis: entirely NEW, or terminal GROWTH + PERSISTENT + terminal GROWTH.

Growth must connect to an interior previous tip with compatible tangent and
unmatched terminal evidence. NEW and GROWTH cannot coexist on one accepted axis.
"""
 xy=current['xy'];emission=_emission(probability,current);labels=np.zeros(len(xy),np.uint8);records=[];h,w=current['axis_map'].shape
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
     if boundary>=3 and boundary<n-5 and current['endkinds'][ai-1][0]=='tip' and score[:boundary].mean()<.35 and float((score[boundary:]>.4).mean())>.15:prefix.append(boundary)
     if n-boundary>=3 and boundary>5 and current['endkinds'][ai-1][1]=='tip' and score[boundary:].mean()<.35 and float((score[:boundary]>.4).mean())>.15:suffix.append(boundary)
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
  labels[path]=chosen;records.append(dict(**info,mode=mode,growth_nodes=int((chosen==3).sum()),persistent_nodes=int((chosen==1).sum()),new_nodes=int((chosen==2).sum()),new_and_growth_on_same_axis=False,decision_margin=float(margin) if margin is not None else None))
 seed=np.zeros(current['axis_map'].shape,bool);seed[tuple(xy.T)]=True;_,near=ndi.distance_transform_edt(~seed,return_indices=True);index=np.zeros(seed.shape,np.int32);index[tuple(xy.T)]=np.arange(len(xy));prediction=labels[index[tuple(near)]];mask=current['axis_map']>0;prediction[~mask]=0
 # Where a nearby different axis owns the width, use its nearest own skeleton.
 for ai,path in enumerate(current['paths'],1):
  region=current['axis_map']==ai
  if not region.any():continue
  ry,rx=np.where(region);_,nn=cKDTree(xy[path]).query(np.stack([ry,rx],1));prediction[ry,rx]=labels[path[nn]]
 assert np.array_equal(prediction>0,mask)
 assert all(not (v['new_nodes'] and v['growth_nodes']) for v in records)
 return prediction,records

def save(path,axis,link=None):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=dict(xy=axis['xy'],axis_map=axis['axis_map'],uncertain=axis['uncertain']);payload.update({f'path_{i}':v for i,v in enumerate(axis['paths'])});np.savez_compressed(path,**payload)
 meta=dict(endkinds=axis['endkinds'],decisions=axis['decisions'],putative_geometry_not_confirmed_root_identity=True)
 if link:meta['axis_matches']=link['axis_info']
 path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
