"""Matched internal evaluation, full-field categorical outputs and seeded IDs."""
from pathlib import Path
import json,csv,sys,math
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from prepare import O,R,BASE,PERIODS,MODELS,js,csvout,sha
from metrics import binary_curve,adjusted_rand
from train import link_report
CLASSES={1:'persistent',2:'newly_observed',3:'tip_extension'}

def class_metrics(cm):
 out=[]
 for k,name in CLASSES.items():
  tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp)
  out.append(dict(class_code=k,class_name=name,Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),Dice=2*tp/max(2*tp+fp+fn,1),IoU=tp/max(tp+fp+fn,1),TP=tp,FP=fp,FN=fn,reference_pixels=int(cm[k].sum())))
 return out

class DSU:
 def __init__(self,n):self.p=np.arange(n)
 def find(self,a):
  while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
  return a
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)

def tracks(model,kind,nodes,g,geo,source,area):
 print('TRACKS_BEGIN',model,kind,flush=True);out=O/(model+'_'+kind);prediction=np.load(out/'inference.npz');p=prediction['relation_prob'][:,1];threshold=json.loads((out/'completion.json').read_text())['selected_validation_links']['threshold'];pairs=g['pairs'];dsu=DSU(len(nodes));segments={(f['period'],s['id']):s for f in source['root_structure_annotation']['periods'] for s in f['segments']}
 for (a,b),native in zip(pairs,g['native_adjacency']):
  if native:dsu.union(int(a),int(b))
 # Sparse route matching: one connection per native endpoint. No human root ID,
 # parent relation, temporal class, or manual node connection enters this step.
 best={};spatial=[]
 for i,(a,b) in enumerate(pairs):
  na,nb=nodes[a],nodes[b]
  if p[i]<threshold or na['period']!=nb['period'] or na['segment_id']==nb['segment_id']:continue
  sa=segments[na['period'],na['segment_id']];sb=segments[nb['period'],nb['segment_id']];aa=[];bb=[]
  if na['chunk_start']==0:aa.append((0,np.array(sa['points'][0])))
  if na['chunk_end']==len(sa['points']):aa.append((1,np.array(sa['points'][-1])))
  if nb['chunk_start']==0:bb.append((0,np.array(sb['points'][0])))
  if nb['chunk_end']==len(sb['points']):bb.append((1,np.array(sb['points'][-1])))
  candidates=[(np.linalg.norm(pa-pb),ea,eb) for ea,pa in aa for eb,pb in bb]
  if not candidates:continue
  distance,ea,eb=min(candidates)
  if distance>18:continue
  ka=(na['period'],na['segment_id'],ea);kb=(nb['period'],nb['segment_id'],eb);score=float(p[i]-.001*distance);spatial.append((i,ka,kb,score))
  for key in [ka,kb]:
   if key not in best or score>best[key][1]:best[key]=(i,score)
 for i,a,b,_ in spatial:
  if best[a][0]==i and best[b][0]==i:dsu.union(*map(int,pairs[i]))
 # Mutual-best adjacent-date correspondence, threshold fixed on validation.
 best={};temporal=[]
 for i,(a,b) in enumerate(pairs):
  na,nb=nodes[a],nodes[b]
  if p[i]<threshold or abs(na['period_index']-nb['period_index'])!=1:continue
  score=float(p[i]-.0005*g['edge_attr'][i,1]);ka=(int(a),nb['period_index']);kb=(int(b),na['period_index']);temporal.append((i,ka,kb))
  for key in [ka,kb]:
   if key not in best or score>best[key][1]:best[key]=(i,score)
 for i,a,b in temporal:
  if best[a][0]==i and best[b][0]==i:dsu.union(*map(int,pairs[i]))
 _,cluster=np.unique([dsu.find(i) for i in range(len(nodes))],return_inverse=True);nclusters=int(cluster.max()+1);confirmed={h['root_id']:h for h in source['root_structure_annotation']['root_track_hierarchy'] if h.get('annotation_status_v60')=='user_defined'};rids=sorted(confirmed);birth={rid:min(n['period_index'] for n in nodes if n['root_target']==rid and area[n['node']]>0) if any(n['root_target']==rid and area[n['node']]>0 for n in nodes) else min(n['period_index'] for n in nodes if n['root_target']==rid) for rid in rids};seed=np.zeros((nclusters,len(rids)),np.int64);ri={rid:i for i,rid in enumerate(rids)}
 for n in nodes:
  rid=n['root_target']
  if rid in birth and n['period_index']==birth[rid]:seed[cluster[n['node']],ri[rid]]+=int(area[n['node']])
 cc,rr=linear_sum_assignment(-seed);assigned={}
 for c,r in zip(cc,rr):
  if seed[c,r]>0 and seed[c,r]/max(seed[c].sum(),1)>=.55:assigned[int(c)]=confirmed[rids[r]]['label']
 names={c:assigned.get(c,'P-'+str(c+1).zfill(4)) for c in range(nclusters)};metadata=[]
 for c in range(nclusters):
  s=seed[c];present=np.flatnonzero(s>0);metadata.append(dict(track=c,track_key='P-'+str(c+1).zfill(4),display_label=names[c],seeded_human_name=c in assigned,seed_label_conflict=bool(len(present)>1),anchor_purity=float(s.max()/max(s.sum(),1)),anchor_labels=[confirmed[rids[r]]['label'] for r in present],nodes=int((cluster==c).sum())))
 print('TRACKS_CLUSTERED',model,kind,nclusters,flush=True);np.save(out/'node_track.npy',cluster);js(out/'track_registry.json',metadata)
 rows=[];test_eval=[];switches=[]
 for rid,h in confirmed.items():
  prev=None;switch=0
  for ti,t in enumerate(PERIODS):
   ids=np.array([n['node'] for n in nodes if n['root_target']==rid and n['period']==t],int);observed=bool(len(ids) and area[ids].sum()>0);dominant=None;coverage=0;parts=0
   if observed:
    counts=np.bincount(cluster[ids],weights=area[ids],minlength=nclusters);dominant=int(counts.argmax());coverage=float(counts.max()/max(counts.sum(),1));parts=int((counts>0).sum());switch+=int(prev is not None and prev!=dominant);prev=dominant
   split=next((int(g['node_split'][n['node']]) for n in nodes if n['root_target']==rid),-1);seed_period=ti==birth[rid]
   row=dict(model=model,variant=kind,split=['train','val','test'][split],period=t,human_label=h['label'],human_parent_label=confirmed.get(h.get('parent_root_id'),{}).get('label',''),observed=observed,first_observation_seed=seed_period,predicted_track='' if dominant is None else 'P-'+str(dominant+1).zfill(4),predicted_display_label='' if dominant is None else names[dominant],display_name_matches=bool(observed and names[dominant]==h['label']),dominant_track_pixel_fraction=coverage,predicted_fragments=parts,seed_label_conflict=False if dominant is None else metadata[dominant]['seed_label_conflict'])
   rows.append(row)
   if split==2 and observed and not seed_period:test_eval.append(row)
  if split==2:switches.append(switch)
 csvout(out/'Root_ID_comparison.csv',rows)
 held=np.array([n['node'] for n in nodes if g['node_split'][n['node']]==2 and n['root_target'] and n['period_index']>birth[n['root_target']] and area[n['node']]>0]);true=[];pred=[]
 for nid in held:
  n=nodes[nid];repeat=int(area[nid]);true.extend([n['root_target']]*repeat);pred.extend([int(cluster[nid])]*repeat)
 result=dict(scope='Held-out manual families; first-observation seed names excluded from label agreement',canonical_numbers_are_seed_names_not_network_class_logits=True,first_observation_labels_not_used_as_neural_features=True,total_predicted_tracks=nclusters,heldout_post_seed_root_dates=len(test_eval),heldout_post_seed_name_accuracy=float(np.mean([r['display_name_matches'] for r in test_eval])) if test_eval else None,heldout_post_seed_mean_dominant_fraction=float(np.mean([r['dominant_track_pixel_fraction'] for r in test_eval])) if test_eval else None,heldout_mean_fragments=float(np.mean([r['predicted_fragments'] for r in test_eval])) if test_eval else None,heldout_identity_switches=int(sum(switches)),heldout_pixel_adjusted_Rand=adjusted_rand(true,pred) if true else None,selected_validation_pair_threshold=float(threshold),test_pair=json.loads((out/'test_identity_metrics.json').read_text()))
 js(out/'test_tracking_metrics.json',result);return rows,result

def main():
 if (O/'evaluation_complete.json').exists():print('EVALUATION_ALREADY_COMPLETE');return
 assert (O/'training_complete.json').exists(),'All six new heads must finish before final test evaluation'
 source=json.loads((O/'data/source_progress_17.json').read_text());nodes=json.loads((O/'data/nodes.json').read_text());geo=np.load(O/'data/geometry.npz');g={k:v for k,v in np.load(O/'data/graph.npz').items()};prep=json.loads((O/'data/preparation.json').read_text());area=np.zeros(len(nodes),np.int64)
 regions={};gts={}
 for t in PERIODS:
  nm=geo['node_map_'+t];live=nm>=0;area+=np.bincount(nm[live],minlength=len(nodes));gt=geo['gt_'+t];gts[t]=gt;split=np.full(nm.shape,-1,np.int8);split[live]=g['node_split'][nm[live]];mask=(split==2)&(gt>=1)&(gt<=3);region=binary_dilation(mask,iterations=6)&((gt==0)|mask);regions[t]=region
 table=[];macro=[];all_comparisons=[];id_summary=[]
 for model in MODELS:
  delta={kind:np.load(O/(model+'_'+kind)/'inference.npz')['temporal_delta'] for kind in ['mlp','gat']};cms={v:np.zeros((4,4),np.int64) for v in ['archived','mlp','gat']};rocs={v:{k:[] for k in [1,2,3]} for v in cms};rocy=[]
  for t in PERIODS:
   nm=geo['node_map_'+t];original=np.array(Image.open(BASE/model/'full'/f'{t}_classes.png'));gate=original>0;outputs={v:original.copy() for v in cms}
   if t!='T1':
    base=np.load(BASE/model/'full'/f'{t}_probability.npz')['probability'].astype(np.float32);base/=np.maximum(base.sum(0),1e-7);logs=np.log(np.maximum(base,1e-5));q={v:base for v in ['archived']}
    for kind in ['mlp','gat']:
     z=logs.copy();supported=gate&(nm>=0);z[:,supported]+=delta[kind][nm[supported]].T;z-=z.max(0);p=np.exp(z);p/=p.sum(0);q[kind]=p;outputs[kind]=np.where(gate,p.argmax(0)+1,0).astype(np.uint8)
    region=regions[t];gt=gts[t][region];assert np.all(gt<4)
    rocy.extend(gt.tolist())
    for variant in cms:
     pr=outputs[variant][region];cms[variant]+=np.bincount(gt*4+pr,minlength=16).reshape(4,4)
     for k in [1,2,3]:rocs[variant][k].append(np.where(gate[region],q[variant][k-1][region],0))
   for kind in ['mlp','gat']:
    path=O/(model+'_'+kind)/'full'/f'{t}_classes.png';path.parent.mkdir(exist_ok=True);Image.fromarray(outputs[kind]).save(path);assert np.array_equal(outputs[kind]>0,gate),'A frozen root outline changed'
   print('PREDICTED',model,t,flush=True)
  for variant,cm in cms.items():
   rows=class_metrics(cm)
   for r in rows:table.append(dict(model=model,variant=variant,**r))
   m=dict(model=model,variant=variant,macro_Dice=float(np.mean([r['Dice'] for r in rows])),macro_IoU=float(np.mean([r['IoU'] for r in rows])),accuracy_including_local_background=float(np.trace(cm)/max(cm.sum(),1)),reference='Uploaded17 temporal colors on held-out manually numbered families; local background buffers',confusion=cm.tolist());macro.append(m)
   dest=O/(model+'_'+variant);dest.mkdir(exist_ok=True);js(dest/'test_temporal_metrics.json',dict(**m,classes=rows));np.save(dest/'test_confusion.npy',cm)
   aucs=[]
   for k in [1,2,3]:
    curve=binary_curve(np.array(rocy)==k,np.concatenate(rocs[variant][k]));auc=curve.pop('AUC');ap=curve.pop('average_precision');np.savez_compressed(dest/f'ROC_{k}.npz',**curve);aucs.append(dict(class_code=k,AUC=auc,average_precision=ap))
   js(dest/'test_ROC.json',aucs)
  for kind in ['mlp','gat']:
   rows,res=tracks(model,kind,nodes,g,geo,source,area);all_comparisons.extend(rows);id_summary.append(dict(model=model,variant=kind,heldout_post_seed_root_dates=res['heldout_post_seed_root_dates'],heldout_post_seed_name_accuracy=res['heldout_post_seed_name_accuracy'],heldout_mean_fragments=res['heldout_mean_fragments'],heldout_identity_switches=res['heldout_identity_switches'],heldout_pixel_adjusted_Rand=res['heldout_pixel_adjusted_Rand'],**res['test_pair']))
  print('EVALUATED',model,flush=True)
 csvout(O/'Temporal_class_metrics_before_after.csv',table);csvout(O/'Temporal_macro_metrics_before_after.csv',[{k:v for k,v in m.items() if k!='confusion'} for m in macro]);csvout(O/'Root_ID_comparison_all_models.csv',all_comparisons);csvout(O/'Root_ID_tracking_metrics.csv',id_summary);js(O/'metrics_summary.json',dict(temporal_classes=table,temporal_macro=macro,identity=id_summary))
 assert all(sha(Path(i['path']))==i['SHA256'] for i in prep['immutable_inputs'])
 js(O/'evaluation_complete.json',dict(state='complete',all_six_models_100_epochs=True,source_upload_unchanged=True,old_backbones_heads_predictions_unchanged=True,all_new_binary_root_supports_exactly_preserved=True,test_families=26,identity_seed_names_excluded_from_post_seed_accuracy=True,old_accuracy_numbers_not_reused_as_new_split_results=True));print('EVALUATION_COMPLETE',flush=True)
if __name__=='__main__':main()
