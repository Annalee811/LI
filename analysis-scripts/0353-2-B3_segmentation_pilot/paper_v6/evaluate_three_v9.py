#!/usr/bin/env python3
"""Matched three-model connection fitting, validation selection and strict audit."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import protocol as p
from continuity_v8 import proposals,apply_links,fit_classifier,FEATURES,read
from evaluate_continuity_v8 import anchors,topology,length_field,agreement
from deliver_continuity_v8 import svg
NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}

def result(prob,rgb,threshold,candidates,cfg,classifier):
 base=prob>=threshold
 linked,added,links=apply_links(base,candidates,cfg,classifier) if cfg else (base,np.zeros_like(base),[])
 area=cfg.get('compact_area',0) if cfg else 0;delta=cfg.get('hysteresis_delta',0) if cfg else 0
 grown=p.clean(prob,dict(threshold=threshold,hysteresis_delta=delta,compact_area=0)) if delta else base
 cleaned=p.clean(prob,dict(threshold=threshold,hysteresis_delta=delta,compact_area=area)) if area else grown
 combined=linked|grown;removed=grown&~cleaned
 # Never remove any member of a connected object that has a new bridge.
 labels,_=ndi.label(combined,np.ones((3,3),np.uint8));protected=np.unique(labels[added]);protected=protected[protected>0]
 removed&=~np.isin(labels,protected);out=combined&~removed
 return out,out&~base,base&~out,links

def partition(data,folder,col,threshold):
 maps={t:np.load(folder/(t+'_probability.npy')) for t in p.PERIODS};candidates={};refs={}
 for t in p.PERIODS:
  x=p.COLS[col];_,candidates[t]=proposals(maps[t],data.rgb[t][:,x:x+448],threshold);refs[t]=anchors(data.gt[t][:,x:x+448]==255)
 return maps,candidates,refs

def score(data,maps,col,threshold,candidates,cfg,classifier,refs):
 total=np.zeros(4,np.int64);per={};topos=[];stats={k:0 for k in ['linked_reference_pairs','reference_pairs','components','skeleton_endpoints','accepted_links','added_pixels','removed_pixels','added_reference_root_pixels','added_reference_background_pixels','removed_reference_root_pixels','removed_reference_background_pixels']}
 for t in p.PERIODS:
  x=p.COLS[col];gt=data.gt[t][:,x:x+448];g=gt==255;mask,added,removed,links=result(maps[t],data.rgb[t][:,x:x+448],threshold,candidates[t],cfg,classifier)
  cc=p.counts(gt,mask);total+=cc;per[t]=p.metrics(cc);tm=topology(g,mask,refs[t]);topos.append(tm['clDice'])
  for k in ['linked_reference_pairs','reference_pairs','components','skeleton_endpoints']:stats[k]+=tm[k]
  stats['accepted_links']+=len(links)
  for prefix,im in [('added',added),('removed',removed)]:
   stats[prefix+'_pixels']+=int(im.sum());stats[prefix+'_reference_root_pixels']+=int((im&g).sum());stats[prefix+'_reference_background_pixels']+=int((im&~g).sum())
 return dict(root=p.metrics(total),per_period=per,topology=dict(**stats,clDice_mean=float(np.mean(topos)),reference_pair_connectivity=stats['linked_reference_pairs']/max(1,stats['reference_pairs'])))

def fit(data,out,threshold):
 feats=[];targets=[];audit=[]
 for col in range(3):
  for t in p.PERIODS:
   prob=np.load(out/'probabilities/train'/('C'+str(col))/(t+'_probability.npy'));x=p.COLS[col];gt=data.gt[t][:,x:x+448]==255;rgb=data.rgb[t][:,x:x+448];_,cc=proposals(prob,rgb,threshold)
   labels,_=ndi.label(gt,np.ones((3,3),np.uint8));dist,nearest=ndi.distance_transform_edt(~gt,return_indices=True);pos=neg=amb=0
   for c in cc:
    yy,xx=np.array(c['pixels']).T;cover=float((dist[yy,xx]<=1).mean());a,b=np.array(c['endpoints']);same=dist[tuple(a)]<=1 and dist[tuple(b)]<=1 and labels[tuple(nearest[:,a[0],a[1]])]==labels[tuple(nearest[:,b[0],b[1]])]
    target=1 if cover>=.9 and same else (0 if cover<=.2 or not same else None)
    if target is None:amb+=1;continue
    feats.append(c['features']);targets.append(target);pos+=target;neg+=1-target
   audit.append(dict(period=t,column=col,positive=pos,negative=neg,ambiguous_excluded=amb,proposals=len(cc)))
 classifier=fit_classifier(feats,targets);p.js(out/'link_classifier.json',classifier);p.csvout(out/'link_training_audit.csv',audit);np.savez_compressed(out/'link_training_features.npz',features=np.array(feats),targets=np.array(targets),names=np.array(FEATURES))
 return classifier

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--select-only',action='store_true');a.add_argument('--frozen',action='store_true');args=a.parse_args();data=p.Data(args.data);run=args.run;cache={};selections={}
 for name in NAMES:
  out=run/name;threshold=read(out/'selection.json')['threshold']
  if args.frozen:
   classifier=read(out/'link_classifier.json');maps,cc,refs=partition(data,out/'probabilities/val',3,threshold);selections[name]=read(out/'postprocessing_selection.json');cache[name]=(classifier,maps,cc,refs);continue
  classifier=fit(data,out,threshold);maps,cc,refs=partition(data,out/'probabilities/val',3,threshold)
  base=score(data,maps,3,threshold,cc,None,classifier,refs);best=base;chosen=None;trials=[]
  configs=[dict(max_gap=gap,min_score=minimum,min_cosine=.5,link_threshold=th,compact_area=area,hysteresis_delta=delta) for gap,minimum,th,area,delta in itertools.product([10,18],[.2,.3],[.35,.5,.65],[0,8,20],[0,.025,.05])]
  # A pure cleanup option is also allowed; baseline always remains eligible.
  configs += [dict(max_gap=0,min_score=.3,min_cosine=.5,link_threshold=1.,compact_area=area,hysteresis_delta=delta) for area,delta in itertools.product([0,8,20],[0,.025,.05]) if area or delta]
  for cfg in configs:
   sc=score(data,maps,3,threshold,cc,cfg,classifier,refs);ok=sc['root']['Recall']>=base['root']['Recall']-.002 and sc['root']['Precision']>=base['root']['Precision']-.002 and sc['topology']['reference_pair_connectivity']>=base['topology']['reference_pair_connectivity']-.001
   trials.append(dict(**cfg,admissible=ok,**sc['root'],**sc['topology']))
   rank=lambda s:(s['root']['Dice'],s['topology']['reference_pair_connectivity'],-s['topology']['removed_pixels'])
   if ok and rank(sc)>rank(best):chosen=cfg;best=sc
  p.csvout(out/'postprocessing_validation_grid.csv',trials);selection=dict(threshold=threshold,configuration=chosen,before=base,after=best,selection_uses_C4=False,criterion='Maximum strict validation Dice; recall/precision may fall at most0.002, reference-pair connectivity at most0.001; baseline remains eligible')
  p.js(out/'postprocessing_selection.json',selection);selections[name]=selection;cache[name]=(classifier,maps,cc,refs)
  print('V9_POSTPROCESSING_SELECTED',name,base['root']['Dice'],best['root']['Dice'],chosen,flush=True)
 winner=max(NAMES,key=lambda n:selections[n]['after']['root']['Dice'])
 if not args.frozen:p.js(run/'frozen_selection.json',dict(winner=winner,selection_uses_C4=False,models=selections))
 else:assert read(run/'frozen_selection.json')['models']==selections
 if args.select_only:return
 reports={};rows=[];pairs=[];lengthstats={};whole={};smooth={}
 for name in NAMES:
  out=run/name;sel=selections[name];threshold=sel['threshold'];cfg=sel['configuration'];classifier,maps,cc,refs=cache[name];reports[name]={};lengthstats[name]={}
  for split,col in [('val',3),('test',4)]:
   if col==4:maps,cc,refs=partition(data,out/'probabilities/test/C4',col,threshold)
   before=score(data,maps,col,threshold,cc,None,classifier,refs);after=score(data,maps,col,threshold,cc,cfg,classifier,refs);reports[name][split]=dict(before=before,after=after);lengthstats[name][split]={}
   for mode,sc in [('before',before),('after',after)]:
    rows.append(dict(model=name,partition=split,mode=mode,period='all',**sc['root'],**sc['topology']))
   folder=out/'evaluation'/split;folder.mkdir(parents=True,exist_ok=True)
   for t in p.PERIODS:
    x=p.COLS[col];gt=data.gt[t][:,x:x+448]==255;rgb=data.rgb[t][:,x:x+448];mask,added,removed,links=result(maps[t],rgb,threshold,cc[t],cfg,classifier)
    beforemask=maps[t]>=threshold
    for label,im in [('human',gt),('before',beforemask),('after',mask),('added',added),('removed',removed)]:Image.fromarray(im.astype(np.uint8)*255).save(folder/(t+'_'+label+'.png'))
    p.js(folder/(t+'_links.json'),links);lf=length_field(gt)
    for mode,im in [('before',beforemask),('after',mask)]:
     lp=length_field(im)
     for yi in range(4):
      for xi in range(2):
       sl=(slice(yi*277,(yi+1)*277),slice(xi*224,(xi+1)*224));pairs.append(dict(model=name,partition=split,mode=mode,period=t,block='X%dY%d'%(xi,yi),human_length_px=float(lf[sl].sum()),predicted_length_px=float(lp[sl].sum())))
   for mode in ['before','after']:
    rr=[r for r in pairs if r['model']==name and r['partition']==split and r['mode']==mode];lengthstats[name][split][mode]=agreement([r['human_length_px'] for r in rr],[r['predicted_length_px'] for r in rr])
   print('V9_METRICS',name,split,after['root'],flush=True)
  images=out/'images';images.mkdir(exist_ok=True);whole[name]={};smooth[name]={}
  for t in p.PERIODS:
   rgb=data.rgb[t];prob=np.load(out/'probabilities/upper'/(t+'_probability.npy'));base,cc=proposals(prob,rgb,threshold);mask,added,removed,links=result(prob,rgb,threshold,cc,cfg,classifier)
   for label,im in [('root_mask',mask),('before_mask',base),('added',added),('removed',removed)]:Image.fromarray(im.astype(np.uint8)*255).save(images/(t+'_'+label+'.png'))
   Image.fromarray(np.round(np.clip(prob,0,1)*65535).astype(np.uint16)).save(images/(t+'_probability_uint16.png'))
   rgba=np.concatenate([rgb,mask[:,:,None].astype(np.uint8)*255],axis=2);rgba[~mask,:3]=0;Image.fromarray(rgba).save(images/(t+'_roots_RGBA.png'))
   overlay=rgb.copy();overlay[mask]=(rgb[mask]*.55+np.array([0,133,125])*.45).astype(np.uint8);overlay[added]=(rgb[added]*.5+np.array([213,94,0])*.5).astype(np.uint8);Image.fromarray(overlay).save(images/(t+'_overlay.png'))
   # RGB values remain unchanged in the standalone source and extracted roots.
   assert np.array_equal(rgba[mask,:3],rgb[mask]);smooth[name][t]=svg(mask,images/(t+'_centerline_display.svg'));p.js(images/(t+'_links.json'),links);whole[name][t]=dict(links=len(links),added_pixels=int(added.sum()),removed_pixels=int(removed.sum()))
  print('V9_PRODUCTS',name,whole[name],flush=True)
 p.js(run/'all_metrics.json',reports);p.csvout(run/'metrics.csv',rows);p.csvout(run/'root_length_pairs.csv',pairs);p.js(run/'root_length_agreement.json',lengthstats);p.js(run/'whole_field_changes.json',whole);p.js(run/'centerline_display_verification.json',smooth)
 p.js(run/'verification.json',dict(passed=True,manual_progress_SHA256=p.sha(args.data/'original_human_progress.json'),source_provenance_SHA256=p.sha(args.data/'dataset/provenance.json'),ground_truth_changed=False,death_or_temporal_classes_trained=False,external_independent_test_images=0,whole_images_qualitative=True))
if __name__=='__main__':main()
