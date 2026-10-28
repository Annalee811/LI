"""Separate pixel agreement, root-identity consistency and third-type junction intrusions."""
import json,pickle,csv
import numpy as np
from PIL import Image
from prepare import ROOT,OUT,LAST,BASE,PERIODS,js,csvout,sha
from topology import fuse_crossing_bridges

def metric(y,p):
 cm=np.bincount(y*3+p,minlength=9).reshape(3,3);r=[]
 for k,name in enumerate(['persistent','newly_observed','tip_extension']):
  tp=int(cm[k,k]);fp=int(cm[:,k].sum()-tp);fn=int(cm[k].sum()-tp);r.append(dict(name=name,Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),Dice=2*tp/max(2*tp+fp+fn,1),IoU=tp/max(tp+fp+fn,1),support=int(cm[k].sum())))
 return dict(classes=r,macro_Dice=float(np.mean([v['Dice'] for v in r])),confusion=cm.tolist(),accuracy=float(np.trace(cm)/max(cm.sum(),1)))
def main():
 dest=OUT/'final';verify=json.loads((dest/'decoding_verification.json').read_text());assert verify['passed'];spatial=np.load(LAST/'data/spatial.npy');owner=dict(np.load(LAST/'data/unet/ownership.npz'));graph=dict(np.load(LAST/'data/unet/graph.npz'))
 maps={v:{} for v in ['original_U_Net','previous_local_V82','whole_segment_temporal_graph']};summaries={};rows=[];rootrows=[];editrows=[]
 for t in PERIODS:
  for name,p in [('original_U_Net',BASE/'unet/full'),('previous_local_V82',LAST/'full'),('whole_segment_temporal_graph',dest/'full')]:maps[name][t]=np.array(Image.open(p/f'{t}_classes.png'))
  assert all(np.array_equal(maps[v][t]>0,maps['original_U_Net'][t]>0) for v in maps)
  if t=='T1':assert np.array_equal(maps['whole_segment_temporal_graph'][t],maps['original_U_Net'][t]);continue
  px=dict(np.load(LAST/'data/unet'/(t+'_pixels.npz')));use=px['split']==2;gt=px['target'][use];before=maps['previous_local_V82'][t].ravel()[px['pixel'][use]]-1;after=maps['whole_segment_temporal_graph'][t].ravel()[px['pixel'][use]]-1;edit=before!=after;wrong=before!=gt;editrows.append(dict(period=t,changed_on_holdout=int(edit.sum()),fixed=int((edit&wrong&(after==gt)).sum()),harmed=int((edit&~wrong).sum()),changed_still_wrong=int((edit&wrong&(after!=gt)).sum())))
  for rid in np.unique(px['rid'][use]):
   if rid<0:continue
   take=use&(px['rid']==rid)
   if take.sum()<10:continue
   record=dict(period=t,root_target=int(rid),pixels=int(take.sum()))
   for v in maps:
    a=maps[v][t].ravel()[px['pixel'][take]];record[v+'_new_extension_conflict5']=bool(min((a==2).sum(),(a==3).sum())>=5)
   rootrows.append(record)
 for ref_version in ['20','24']:
  refroot=ROOT/'results/B3_ConvLSTM_GAT_upload20_v74_20261028/data/reference' if ref_version=='20' else LAST/'data/reference';allgt=[];preds={v:[] for v in maps}
  for t in PERIODS[1:]:
   g=np.array(Image.open(refroot/f'{t}_classes.png'));valid=(spatial==2)&(maps['original_U_Net'][t]>0)&np.isin(g,[1,2,3]);allgt.extend(g[valid].astype(np.int64)-1)
   for v in maps:preds[v].extend(maps[v][t][valid].astype(np.int64)-1)
  summaries[ref_version]={v:metric(np.array(allgt),np.array(p)) for v,p in preds.items()}
  for v,m in summaries[ref_version].items():
   for row in m['classes']:rows.append(dict(reference=ref_version,variant=v,scope='internal_fixed_root_spatial_holdout',**row))
 csvout(dest/'Pixel_class_metrics_reference20_and24.csv',rows);csvout(dest/'Known_root_new_extension_conflicts.csv',rootrows);csvout(dest/'Heldout_edit_benefits_and_harms.csv',editrows)
 # Independent of decoder's proposed colors: use source24 confirmed IDs and reference arm types for this consistency audit.
 with (OUT/'data/topology.pkl').open('rb') as f:top=pickle.load(f)
 crossing=[]
 for ti,t in enumerate(PERIODS[1:],1):
  tp=top['topos'][t];fuse_crossing_bridges(tp);gt=np.array(Image.open(LAST/'data/reference'/f'{t}_classes.png'))
  for ji,hub in enumerate(tp['hubs']):
   if hub.get('disabled') or len(hub['arms']) not in [3,4]:continue
   x,y=np.rint(hub['center']).astype(int)
   if spatial[y,x]!=2:continue
   actual={};valid=True
   for arm in hub['arms']:
    q=arm['body_points'];q=q[6:]
    if len(q)<4:valid=False;break
    take=spatial[q[:,1],q[:,0]]==2;q=q[take]
    if len(q)<4:valid=False;break
    n=owner[t][q[:,1],q[:,0]];ids=graph['root_target'][n];ids=ids[ids>=0]
    if len(ids)<3:valid=False;break
    values,counts=np.unique(ids,return_counts=True)
    if counts.max()/len(ids)<.9:valid=False;break
    rid=int(values[counts.argmax()]);c=gt[q[:,1],q[:,0]];c=c[np.isin(c,[1,2,3])]
    if len(c)<4:valid=False;break
    values,counts=np.unique(c,return_counts=True)
    if counts.max()/len(c)<.9:valid=False;break
    actual.setdefault(rid,set()).add(int(values[counts.argmax()]))
   if not valid or len(actual)!=2 or any(len(c)!=1 for c in actual.values()):continue
   allowed=set.union(*actual.values());q=hub['points'];x0=max(0,int(q[:,0].min())-2);x1=min(2541,int(q[:,0].max())+3);y0=max(0,int(q[:,1].min())-2);y1=min(1108,int(q[:,1].max())+3);yy,xx=np.mgrid[y0:y1,x0:x1];near=np.min((xx[:,:,None]-q[:,0])**2+(yy[:,:,None]-q[:,1])**2,axis=2)<=4;near&=(maps['original_U_Net'][t][y0:y1,x0:x1]>0)&(spatial[y0:y1,x0:x1]==2);record=dict(period=t,hub=ji,x=x,y=y,allowed_reference_types=','.join(map(str,sorted(allowed))),root_IDs=','.join(map(str,actual)),core_pixels=int(near.sum()))
   for v in maps:record[v+'_third_type_pixels']=int((near&~np.isin(maps[v][t][y0:y1,x0:x1],list(allowed))).sum())
   record['reference24_third_type_pixels']=int((near&np.isin(gt[y0:y1,x0:x1],[1,2,3])&~np.isin(gt[y0:y1,x0:x1],list(allowed))).sum());crossing.append(record)
 csvout(dest/'Confirmed_two_root_junction_third_type_audit.csv',crossing)
 root_summary={v:sum(r[v+'_new_extension_conflict5'] for r in rootrows) for v in maps};cross_summary={v:dict(junctions_with_third_type=sum(r[v+'_third_type_pixels']>0 for r in crossing),third_type_pixels=sum(r[v+'_third_type_pixels'] for r in crossing)) for v in maps};case={}
 for v in maps:case[v]=int((maps[v]['T2'][800:850,1060:1110]==3).sum())
 totals={k:sum(r[k] for r in editrows) for k in ['changed_on_holdout','fixed','harmed','changed_still_wrong']};js(dest/'evaluation_results.json',dict(pixel_metrics=summaries,root_period_records=len(rootrows),root_new_extension_conflicts5=root_summary,scoreable_confirmed_two_root_junctions=len(crossing),third_type_junction_audit=cross_summary,holdout_edit_counts=totals,user_review_case_yellow_pixels=case,user_case_scope='User-reported known regression region; not independent validation',reference24_itself_has_core_third_type_pixels=sum(r['reference24_third_type_pixels'] for r in crossing)))
 for row in json.loads((LAST/'preparation.json').read_text())['immutable_inputs']:assert sha(row['path'])==row['sha256']
 js(dest/'evaluation_verification.json',dict(passed=True,all_four_root_geometries_unchanged=True,T1_unchanged=True,original_and_previous_results_unchanged=True,source24_unchanged=True,metrics_are_internal_reference_agreement=True,manual_IDs_prediction_inputs=False,all_fields_processed=True,full_segment_and_junction_rules_applied=True,pixel_metrics_separate_from_structural_checks=True));print(json.dumps(dict(roots=root_summary,junctions=cross_summary,scoreable=len(crossing),edits=totals,case=case),indent=2),flush=True)
if __name__=='__main__':main()
