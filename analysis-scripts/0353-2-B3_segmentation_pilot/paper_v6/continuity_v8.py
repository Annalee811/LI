#!/usr/bin/env python3
"""Short, facing-endpoint root links supported by RGB-derived foreground scores."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.special import expit
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
import torch
import protocol as p
from train import RootUNet,inference

FEATURES=['distance','path_ratio','mean_score','min_score','score_q25','min_facing_cosine','mean_facing_cosine','RGB_local_visibility','RGB_tip_difference','log_min_component_area','target_is_endpoint','target_axis_alignment']

def read(path):return json.loads(Path(path).read_text())
def csvout(path,rows):
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def proposals(prob,rgb,threshold=.55,max_gap=18):
 mask=prob>=threshold;sk=skeletonize(mask);labels,n=ndi.label(mask,np.ones((3,3),np.uint8));area=np.bincount(labels.ravel())
 degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
 ends=np.argwhere(sk&(degree==1));points=[];tangents=[];ids=[]
 h,w=mask.shape
 for point in ends:
  y,x=map(int,point);lab=int(labels[y,x])
  if area[lab]<12:continue
  path=[(y,x)];previous=None;current=(y,x)
  for _ in range(7):
   yy,xx=current;neighbors=[(yy+dy,xx+dx) for dy in [-1,0,1] for dx in [-1,0,1] if (dy or dx) and 0<=yy+dy<h and 0<=xx+dx<w and sk[yy+dy,xx+dx] and (yy+dy,xx+dx)!=previous]
   if len(neighbors)!=1:break
   previous=current;current=neighbors[0];path.append(current)
  if len(path)<4:continue
  vector=point-np.array(path[-1]);length=np.linalg.norm(vector)
  if length<2.5:continue
  points.append(point);tangents.append(vector/length);ids.append(lab)
 if not points:return mask,[]
 points=np.array(points);tangents=np.array(tangents);ids=np.array(ids);tree=cKDTree(points)
 gray=rgb.astype(np.float32).mean(axis=2)/255;detail=abs(ndi.gaussian_filter(gray,.65)-ndi.gaussian_filter(gray,3.))
 mean=ndi.gaussian_filter(gray,5);std=np.sqrt(np.maximum(ndi.gaussian_filter(gray**2,5)-mean**2,0));visibility=detail/(std+.025)
 pair_specs=[]
 for i,j in sorted(tree.query_pairs(max_gap)):
  if ids[i]==ids[j]:continue
  a,b=points[i],points[j];delta=b-a;distance=float(np.linalg.norm(delta))
  if distance<2:continue
  c1=float(np.dot(tangents[i],delta/distance));c2=float(np.dot(tangents[j],-delta/distance))
  if min(c1,c2)<.5:continue
  pair_specs.append((i,j,a,b,int(ids[i]),int(ids[j]),distance,c1,c2,True,abs(c2)))
 # A broken lateral can end near the middle of its parent, not at another tip.
 interior=np.argwhere(sk&(degree>=2));interior_tree=cKDTree(interior) if len(interior) else None
 if interior_tree is not None:
  for i,a in enumerate(points):
   best_by_component={}
   for k in interior_tree.query_ball_point(a,max_gap):
    b=interior[k];lab=int(labels[tuple(b)])
    if lab==ids[i] or area[lab]<12:continue
    delta=b-a;distance=float(np.linalg.norm(delta))
    if distance<2:continue
    c1=float(np.dot(tangents[i],delta/distance))
    if c1<.5:continue
    if lab in best_by_component and best_by_component[lab][0]<=distance:continue
    near=interior[np.array(interior_tree.query_ball_point(b,3))];center=near-near.mean(axis=0)
    if len(near)>=3:
     _,vectors=np.linalg.eigh(center.T@center);alignment=abs(float(np.dot(vectors[:,-1],delta/distance)))
    else:alignment=0.
    best_by_component[lab]=(distance,k,b,c1,alignment)
   for lab,(distance,k,b,c1,alignment) in best_by_component.items():
    pair_specs.append((i,len(points)+k,a,b,int(ids[i]),lab,distance,c1,c1,False,alignment))
 candidates=[]
 for i,j,a,b,id1,id2,distance,c1,c2,target_endpoint,alignment in pair_specs:
  lo=np.maximum(0,np.minimum(a,b)-3);hi=np.minimum([h,w],np.maximum(a,b)+4);sl=(slice(lo[0],hi[0]),slice(lo[1],hi[1]));local=prob[sl]
  cost=(-np.log(np.maximum(local,.01))+.15).astype(np.float64);cost[local<.18]=np.inf
  lab=labels[sl];cost[(lab>0)&(lab!=id1)&(lab!=id2)]=np.inf
  try:path,path_cost=route_through_array(cost,tuple(a-lo),tuple(b-lo),fully_connected=True,geometric=True)
  except (ValueError,RuntimeError):continue
  if not np.isfinite(path_cost):continue
  path=np.array(path,dtype=int)+lo;steps=np.diff(path,axis=0);length=float(np.linalg.norm(steps,axis=1).sum())
  if length>1.5*distance:continue
  new=~mask[path[:,0],path[:,1]];added=path[new]
  if not len(added) or len(added)>24:continue
  yy,xx=added.T;scores=prob[yy,xx];tip=(rgb[a[0],a[1]].astype(float)+rgb[b[0],b[1]].astype(float))/2
  feature=[distance,length/distance,float(scores.mean()),float(scores.min()),float(np.quantile(scores,.25)),min(c1,c2),(c1+c2)/2,
   float(visibility[yy,xx].mean()),float(np.linalg.norm(rgb[yy,xx].mean(axis=0)-tip)/255),float(np.log1p(min(area[id1],area[id2]))),int(target_endpoint),alignment]
  candidates.append(dict(endpoint_ids=[int(i),int(j)],components=[id1,id2],endpoints=[a.tolist(),b.tolist()],pixels=added.tolist(),features=feature,kind='tip_to_tip' if target_endpoint else 'lateral_to_parent'))
 return mask,candidates

def apply_links(mask,candidates,configuration,classifier):
 result=mask.copy();added=np.zeros_like(mask);used=set();parent={};accepted=[]
 owners,_=ndi.label(mask,np.ones((3,3),np.uint8))
 def find(x):
  parent.setdefault(x,x)
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 ordered=[]
 for c in candidates:
  f=c['features'];score=float(expit(np.dot((np.array(f)-classifier['mean'])/classifier['scale'],classifier['weights'])+classifier['intercept'])) if classifier['fitted'] else f[2]
  if f[0]>configuration['max_gap'] or f[3]<configuration['min_score'] or f[5]<configuration['min_cosine']:continue
  if score<configuration['link_threshold']:continue
  ordered.append((score,c))
 for score,c in sorted(ordered,key=lambda v:v[0],reverse=True):
  i,j=c['endpoint_ids'];a,b=c['components']
  if i in used or j in used or find(a)==find(b):continue
  pixels=np.array(c['pixels']);yy,xx=pixels.T;touching=set()
  for y,x in pixels:
   touching.update(owners[max(0,y-1):y+2,max(0,x-1):x+2].ravel())
  touching.discard(0);intended={find(a),find(b)}
  if any(find(int(lab)) not in intended for lab in touching):continue
  result[yy,xx]=True;added[yy,xx]=True;owners[yy,xx]=a
  used.update([i,j]);parent[find(a)]=find(b);accepted.append(dict(**c,link_score=score))
 assert np.all(result[mask]);return result,added,accepted

def fit_classifier(records,targets):
 xx=np.array(records,float);yy=np.array(targets,float);pos=int(yy.sum());neg=int(len(yy)-pos)
 if min(pos,neg)<10:return dict(fitted=False,reason='Fewer than10 clear positive or negative training links; use direct score plus geometry',positive=pos,negative=neg)
 mean=xx.mean(axis=0);scale=xx.std(axis=0);scale[scale<1e-6]=1;x=(xx-mean)/scale;weights=np.where(yy>0,.5/pos,.5/neg)
 def objective(z):
  logits=x@z[:-1]+z[-1];loss=np.sum(weights*(np.logaddexp(0,logits)-yy*logits))+.01*np.sum(z[:-1]**2)
  error=weights*(expit(logits)-yy);gradient=np.r_[x.T@error+.02*z[:-1],error.sum()];return float(loss),gradient
 fit=minimize(objective,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B');assert fit.success
 return dict(fitted=True,features=FEATURES,mean=mean.tolist(),scale=scale.tolist(),weights=fit.x[:-1].tolist(),intercept=float(fit.x[-1]),positive=pos,negative=neg,
  target='Clear training links only: new path pixels within1px of supplied root and both tips on same manual connected root; strict evaluation retains unchanged pixels',training_columns=[0,1,2])

def prepare(args,data):
 run=args.run;run.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model=RootUNet().to(device);model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=True)['state_dict']);infer=inference(model,device,True)
 features=[];targets=[];audit=[]
 for col in range(3):
  folder=run/'train_probabilities'/f'C{col}'
  if not all((folder/(t+'_probability.npy')).exists() for t in p.PERIODS):p.save_maps(folder,data.column(infer,col))
  for t in p.PERIODS:
   prob=np.load(folder/(t+'_probability.npy'));x=p.COLS[col];gt=data.gt[t][:,x:x+448]==255;rgb=data.rgb[t][:,x:x+448];mask,candidates=proposals(prob,rgb)
   labels,_=ndi.label(gt,np.ones((3,3),np.uint8));distance,nearest=ndi.distance_transform_edt(~gt,return_indices=True);clear_pos=clear_neg=ambiguous=0
   for c in candidates:
    pts=np.array(c['pixels']);yy,xx=pts.T;coverage=float((distance[yy,xx]<=1).mean());a,b=np.array(c['endpoints']);same=(distance[tuple(a)]<=1 and distance[tuple(b)]<=1 and labels[tuple(nearest[:,a[0],a[1]])]==labels[tuple(nearest[:,b[0],b[1]])])
    target=1 if coverage>=.9 and same else (0 if coverage<=.2 or not same else None)
    if target is None:ambiguous+=1;continue
    clear_pos+=target;clear_neg+=1-target;features.append(c['features']);targets.append(target)
   audit.append(dict(period=t,column=col,proposals=len(candidates),clear_positive=clear_pos,clear_negative=clear_neg,ambiguous_excluded=ambiguous))
   print('LINK_TEACHER',t,col,len(candidates),clear_pos,clear_neg,ambiguous,flush=True)
 classifier=fit_classifier(features,targets);p.js(run/'link_classifier.json',classifier);csvout(run/'training_link_audit.csv',audit)
 np.savez_compressed(run/'training_link_features.npz',features=np.array(features),targets=np.array(targets),feature_names=np.array(FEATURES))
 p.js(run/'source.json',dict(checkpoint_SHA256=p.sha(args.checkpoint),original_progress_SHA256=p.sha(args.data/'original_human_progress.json'),teacher='User manual training-column root connectivity; C3/C4 excluded from fitting',
  base_model='v6 pretrained U-Net / four-flip TTA',base_threshold=.55,model_retrained=False,link_classifier_trained=classifier['fitted'],gt_used_as_inference_input=False))
 print('LINK_CLASSIFIER_READY',classifier['fitted'],classifier['positive'],classifier['negative'],flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--checkpoint',type=Path,required=True);args=a.parse_args();prepare(args,p.Data(args.data))
if __name__=='__main__':main()
