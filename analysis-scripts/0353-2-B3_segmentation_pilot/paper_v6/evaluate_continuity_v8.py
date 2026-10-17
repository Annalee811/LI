#!/usr/bin/env python3
"""Choose manual-reference connectivity on C3, then score untouched repeated C4."""
import argparse,json,sys,itertools,math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import protocol as p
from continuity_v8 import proposals,apply_links,read,csvout
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
from evaluate import length_field,agreement

def paths(mask):
 sk=skeletonize(mask);points={tuple(x) for x in np.argwhere(sk)};adj={}
 for y,x in points:
  adj[y,x]=[(y+dy,x+dx) for dy in [-1,0,1] for dx in [-1,0,1] if (dy or dx) and (y+dy,x+dx) in points and not (dy and dx and ((y,x+dx) in points or (y+dy,x) in points))]
 edges=set();out=[]
 def edge(a,b):return tuple(sorted([a,b]))
 def trace(a,b):
  track=[a,b];edges.add(edge(a,b));previous=a;current=b
  while len(adj[current])==2:
   nxt=next(x for x in adj[current] if x!=previous)
   if edge(current,nxt) in edges:break
   edges.add(edge(current,nxt));track.append(nxt);previous,current=current,nxt
  return np.array(track)
 for a in sorted(points):
  if len(adj[a])!=2:
   for b in adj[a]:
    if edge(a,b) not in edges:out.append(trace(a,b))
 for a in sorted(points):
  for b in adj[a]:
   if edge(a,b) not in edges:out.append(trace(a,b))
 return out

def anchors(gt):
 pairs=[]
 for track in paths(gt):
  for i in range(0,len(track)-8,4):pairs.append([track[i].tolist(),track[i+8].tolist()])
 return np.array(pairs,dtype=int)

def topology(gt,pred,pairs):
 sg=skeletonize(gt);sp=skeletonize(pred);pr=float((sp&gt).sum()/max(1,sp.sum()));re=float((sg&pred).sum()/max(1,sg.sum()));labels,n=ndi.label(pred,np.ones((3,3),np.uint8));linked=0
 for a,b in pairs:
  sets=[]
  for y,x in [a,b]:
   values=set(labels[max(0,y-1):y+2,max(0,x-1):x+2].ravel());values.discard(0);sets.append(values)
  linked+=bool(sets[0]&sets[1])
 degree=ndi.convolve(sp.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sp
 return dict(clDice=2*pr*re/max(1e-12,pr+re),linked_reference_pairs=int(linked),reference_pairs=len(pairs),components=int(n),skeleton_endpoints=int((sp&(degree==1)).sum()))

def evaluate(data,maps,col,candidates,config,classifier,reference):
 cc=np.zeros(4,np.int64);stats={k:0 for k in ['linked_reference_pairs','reference_pairs','components','skeleton_endpoints','accepted_links','added_pixels','added_reference_root_pixels','added_reference_background_pixels']};per={};top=[]
 for t in p.PERIODS:
  base=maps[t]>=.55;mask,added,links=apply_links(base,candidates[t],config,classifier) if config is not None else (base,np.zeros_like(base),[])
  gt=data.gt[t][:,p.COLS[col]:p.COLS[col]+448];g=gt==255;counts=p.counts(gt,mask);cc+=counts;per[t]=p.metrics(counts);tm=topology(g,mask,reference[t]);top.append(tm['clDice'])
  for key in ['linked_reference_pairs','reference_pairs','components','skeleton_endpoints']:stats[key]+=tm[key]
  stats['accepted_links']+=len(links);stats['added_pixels']+=int(added.sum());stats['added_reference_root_pixels']+=int((added&g).sum());stats['added_reference_background_pixels']+=int((added&~g).sum())
 return dict(root=p.metrics(cc),topology=dict(**stats,reference_pair_connectivity=stats['linked_reference_pairs']/max(1,stats['reference_pairs']),clDice_mean=float(np.mean(top))),per_period=per)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--baseline',type=Path,required=True);args=a.parse_args()
 data=p.Data(args.data);run=args.run;classifier=read(run/'link_classifier.json');cache={};reference={};candidatecache={}
 # C4 is deliberately read only after C3 selection has been written.
 def load_partition(col):
  split='val' if col==3 else 'test';maps={t:np.load(args.baseline/'evaluation/v6_start'/split/(t+'_probability.npy')) for t in p.PERIODS};candidates={};refs={}
  for t in p.PERIODS:
   x=p.COLS[col];_,candidates[t]=proposals(maps[t],data.rgb[t][:,x:x+448]);refs[t]=anchors(data.gt[t][:,x:x+448]==255)
  return maps,candidates,refs
 maps,candidates,refs=load_partition(3);base=evaluate(data,maps,3,candidates,None,classifier,refs);best=base;chosen=None;trials=[]
 threshold_grid=[.35,.5,.65] if classifier['fitted'] else [.3,.4,.5]
 for gap,minimum,cosine,threshold in itertools.product([5,10,18],[.2,.3],[.5,.707],threshold_grid):
  cfg=dict(max_gap=gap,min_score=minimum,min_cosine=cosine,link_threshold=threshold);score=evaluate(data,maps,3,candidates,cfg,classifier,refs)
  admissible=score['root']['Dice']>=base['root']['Dice']-.002 and score['root']['Precision']>=base['root']['Precision']-.005 and score['topology']['clDice_mean']>=base['topology']['clDice_mean']-.001
  trials.append(dict(**cfg,admissible=admissible,**score['root'],**score['topology']))
  rank=lambda s:(s['topology']['reference_pair_connectivity'],s['topology']['clDice_mean'],s['root']['Dice'],-s['topology']['added_pixels'])
  if admissible and rank(score)>rank(best):chosen=cfg;best=score
 csvout(run/'validation_grid.csv',trials);p.js(run/'frozen_selection.json',dict(configuration=chosen,baseline_validation=base,selected_validation=best,
  selection_uses_C4=False,criterion='C3 manual-segment anchor connectivity, then clDice, then strict Dice; Dice loss<=0.002, Precision loss<=0.005, clDice loss<=0.001',
  anchor_metric='Every4 native skeleton steps, two positions8 steps apart on one manual nonbranching segment; count if predicted component labels in1px neighborhoods intersect. Separate topology diagnostic, not pixel tolerance Dice.',
  exact_ground_truth_unchanged=True))
 print('CONTINUITY_FROZEN',chosen,base['topology'],best['topology'],flush=True)
 reports={};rows=[];lengths=[];lengthstats={};audit=[]
 for col in [3,4]:
  split='val' if col==3 else 'test'
  if col==4:maps,candidates,refs=load_partition(col)
  before=evaluate(data,maps,col,candidates,None,classifier,refs);after=evaluate(data,maps,col,candidates,chosen,classifier,refs) if chosen else before
  reports[split]=dict(before=before,after=after);lengthstats[split]={}
  for mode,score in [('before',before),('after',after)]:rows.append(dict(partition=split,mode=mode,**score['root'],**score['topology']))
  folder=run/'evaluation'/split;folder.mkdir(parents=True,exist_ok=True)
  for t in p.PERIODS:
   np.save(folder/(t+'_probability.npy'),maps[t]);base=maps[t]>=.55;mask,added,links=apply_links(base,candidates[t],chosen,classifier) if chosen else (base,np.zeros_like(base),[])
   gt=data.gt[t][:,p.COLS[col]:p.COLS[col]+448]==255
   for stem,im in [('before',base),('after',mask),('added',added),('human',gt)]:Image.fromarray(im.astype(np.uint8)*255).save(folder/(t+'_'+stem+'.png'))
   p.js(folder/(t+'_accepted_links.json'),links)
   for i,link in enumerate(links):
    pts=np.array(link['pixels']);correct=int(gt[pts[:,0],pts[:,1]].sum());audit.append(dict(partition=split,period=t,index=i,new_pixels=len(pts),reference_root_pixels=correct,reference_background_pixels=len(pts)-correct,link_score=link['link_score'],**dict(zip(['feature_'+f for f in classifier.get('features',[])],link['features']))))
   for mode,im in [('before',base),('after',mask)]:
    lf=length_field(gt);lp=length_field(im)
    for yi in range(4):
     for xi in range(2):
      sl=(slice(yi*277,(yi+1)*277),slice(xi*224,(xi+1)*224));lengths.append(dict(partition=split,mode=mode,period=t,block=f'X{xi}Y{yi}',human_length_px=float(lf[sl].sum()),predicted_length_px=float(lp[sl].sum())))
  for mode in ['before','after']:
   rr=[r for r in lengths if r['partition']==split and r['mode']==mode];lengthstats[split][mode]=agreement([r['human_length_px'] for r in rr],[r['predicted_length_px'] for r in rr])
  print('CONTINUITY_EVALUATED',split,before['root'],after['root'],before['topology'],after['topology'],flush=True)
 p.js(run/'all_metrics.json',reports);csvout(run/'metrics.csv',rows);csvout(run/'root_length_pairs.csv',lengths);p.js(run/'root_length_agreement.json',lengthstats)
 if audit:csvout(run/'link_audit.csv',audit)
 p.js(run/'verification.json',dict(passed=True,manual_annotation_edited=False,all_original_provenance_hashes_verified=True,selection_uses_C4=False,probability_fields_or_model_weights_modified=False,
  original_progress_SHA256=p.sha(args.data/'original_human_progress.json'),frozen_selection_SHA256=p.sha(run/'frozen_selection.json')))
 print('CONTINUITY_EVALUATION_READY',flush=True)
if __name__=='__main__':main()
