"""New + existing (persistent + extension) downstream V90 composition.

Uses the existing V90 masks. Measures one binary skeleton per frame, assigns half
each edge length to each endpoint, and only then combines labels. No model fitting.
"""
from pathlib import Path
import argparse,csv,json,hashlib,shutil,sys
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
sys.path.insert(0,'/workspace/LI/analysis-scripts')
from dataset_identity import require_active_position,tube_id
CODE=Path(__file__).resolve().parent
DEFAULT=Path('/workspace/work/V90_two_class_rates_20261123')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rr):
 with p.open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def skeleton_weights(root):
 sk=skeletonize(root,method='zhang');weights=np.zeros(root.shape,np.float64);degree=np.zeros(root.shape,np.uint8)
 # Unique H, V, diagonal-down-right, diagonal-down-left edges. A diagonal
 # is suppressed if either intermediate orthogonal skeleton pixel exists.
 for a,b,orth1,orth2,cost in [
  ((slice(None),slice(None,-1)),(slice(None),slice(1,None)),None,None,1.),
  ((slice(None,-1),slice(None)),(slice(1,None),slice(None)),None,None,1.),
  ((slice(None,-1),slice(None,-1)),(slice(1,None),slice(1,None)),(slice(1,None),slice(None,-1)),(slice(None,-1),slice(1,None)),2**.5),
  ((slice(None,-1),slice(1,None)),(slice(1,None),slice(None,-1)),(slice(1,None),slice(1,None)),(slice(None,-1),slice(None,-1)),2**.5)]:
  edge=sk[a]&sk[b]
  if orth1 is not None:edge &=~sk[orth1]&~sk[orth2]
  weights[a]+=edge*(cost/2);weights[b]+=edge*(cost/2);degree[a]+=edge;degree[b]+=edge
 isolated=sk&(degree==0);weights[isolated]+=1
 assert np.all(weights[~sk]==0)
 return sk,weights,int(isolated.sum())
def summarize(rr,regime,exclude_coarse=False,assigned=False,tube_balanced=False):
 records=[]
 for material in ['0353','A017','Q1']:
  for t in ['T1','T2','T3','T4']:
   q=[r for r in rr if r['material']==material and r['period']==t and not r['baseline']and(not exclude_coarse or not r['coarse_localization'])]
   if regime=='adjacent_only':q=[r for r in q if r['is_adjacent']]
   suffix='assigned'if assigned else'raw'
   q=[r for r in q if r['new_pct_'+suffix]is not None]
   ns=[r['new_pct_'+suffix]for r in q];es=[r['existing_pct_'+suffix]for r in q]
   tubes=sorted(set(r['tube_id']for r in q))
   if tube_balanced:
    ns=[np.mean([r['new_pct_'+suffix]for r in q if r['tube_id']==tb])for tb in tubes]
    es=[np.mean([r['existing_pct_'+suffix]for r in q if r['tube_id']==tb])for tb in tubes]
   result=dict(material=material,period=t,interval_regime=regime,mean_basis='equal_tube_means'if tube_balanced else'equal_position_means',
    classification_support='assigned_only'if assigned else'all_raw_prediction',coarse_position_excluded=exclude_coarse,
    n_positions=len(q),n_tubes=len(tubes),new_mean_pct=float(np.mean(ns))if ns else None,new_SD_pct=float(np.std(ns,ddof=1))if len(ns)>1 else None,
    existing_mean_pct=float(np.mean(es))if es else None,existing_SD_pct=float(np.std(es,ddof=1))if len(es)>1 else None,
    mean_uncertain_skeleton_pct=float(np.mean([r['uncertain_skeleton_pct']for r in q if r['uncertain_skeleton_pct']is not None]))if any(r['uncertain_skeleton_pct']is not None for r in q)else None)
   if ns:
    assert abs(result['new_mean_pct']+result['existing_mean_pct']-100)<1e-8
    if len(ns)>1:assert abs(result['new_SD_pct']-result['existing_SD_pct'])<1e-8
   records.append(result)
 return records
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=DEFAULT);a=ap.parse_args();W=a.output;W.mkdir(parents=True,exist_ok=True)
 shutil.copy2(CODE/'two_class_policy.json',W/'two_class_policy.json');shutil.copy2(Path(__file__),W/'calculate_two_class_rates.py')
 O=Path('/workspace/work/V90_all_microtubes_20261118');active=list(csv.DictReader(Path('/workspace/LI/dataset/current/active_images.csv').open(encoding='utf-8-sig')))
 rows=[];full=[];src_hashes={};maskdir=W/'two_class_native';cache=W/'skeleton_reproduction';cache.mkdir(exist_ok=True)
 for pos in sorted(set(r['position_id']for r in active)):
  assert require_active_position(pos)==pos;rr=sorted([r for r in active if r['position_id']==pos],key=lambda r:r['period']);dest=maskdir/pos;dest.mkdir(parents=True,exist_ok=True);cd=cache/pos;cd.mkdir(exist_ok=True)
  for ix,r in enumerate(rr):
   t=r['period'];base=O/'B3_verified_inference/junction/full'if pos=='0353-2-B3'else O/'delivery/V90_transfer_candidates'/pos
   source=base/(t+'_classes.png');cl=np.array(Image.open(source));assert cl.ndim==2 and set(np.unique(cl)).issubset({0,1,2,3,4})
   source_owner=O/'B3_verified_inference/junction/ownership'/(t+'_axis_owner.npz')if pos=='0353-2-B3'else base/(t+'_ownership.npz')
   owners=dict(np.load(source_owner));unc=owners.get('uncertain');flag_available=unc is not None
   if unc is None:unc=np.zeros(cl.shape,bool)
   root=cl>0;unc=unc&root;sk,w,isolated=skeleton_weights(root);total=float(w.sum());newlen=float(w[cl==2].sum());existing=float(w[(cl==1)|(cl==3)].sum())
   p=float(w[cl==1].sum());e=float(w[cl==3].sum());assert abs(p+e-existing)<1e-8
   baseline=ix==0
   if not baseline:assert abs(newlen+existing-total)<1e-8 and not(cl==4).any()
   else:assert not np.isin(cl,[1,2,3]).any()
   mapped=np.zeros(cl.shape,np.uint8);mapped[np.isin(cl,[1,3])]=1;mapped[cl==2]=2;mapped[cl==4]=255
   assert np.array_equal(mapped==2,cl==2);assert np.array_equal(mapped==1,np.isin(cl,[1,3]))
   Image.fromarray(mapped).save(dest/(t+'_two_classes.png'));quality=mapped.copy();quality[unc]=255;Image.fromarray(quality).save(dest/(t+'_two_classes_uncertainty_retained.png'))
   shutil.copy2(source_owner,dest/(t+'_original_ownership.npz'))
   y,x=np.where(sk);np.savez_compressed(cd/(t+'_skeleton.npz'),y=y.astype(np.int32),x=x.astype(np.int32),weight=w[sk],original_class=cl[sk],uncertain=unc[sk],shape=np.array(cl.shape))
   un=float(w[unc].sum());known=total-un;kn=float(w[(cl==2)&~unc].sum());ke=float(w[np.isin(cl,[1,3])&~unc].sum())
   prev=rr[ix-1]['period']if ix else'';adj=bool(prev)and int(t[1:])-int(prev[1:])==1
   row=dict(position_id=pos,tube_id=tube_id(pos),material=pos.split('-')[0],period=t,previous_observed_period=prev,baseline=baseline,is_adjacent=adj,
    coarse_localization=pos=='Q1-1-Z2',prediction_source='V90_B3_original_automatic'if pos=='0353-2-B3'else'V90_frozen_transfer_adapter',
    total_skeleton_length_px=total,new_skeleton_length_px=None if baseline else newlen,existing_skeleton_length_px=None if baseline else existing,
    new_pct_raw=None if baseline or total==0 else 100*newlen/total,existing_pct_raw=None if baseline or total==0 else 100*existing/total,
    uncertainty_flag_available=flag_available,uncertain_skeleton_pct=100*un/total if flag_available and total>0 else None,
    assigned_skeleton_coverage_pct=100*known/total if flag_available and total>0 else None,
    new_pct_assigned=100*kn/known if flag_available and not baseline and known>0 else None,
    existing_pct_assigned=100*ke/known if flag_available and not baseline and known>0 else None,
    new_pixels=int((cl==2).sum()),existing_pixels=int(np.isin(cl,[1,3]).sum()),isolated_skeleton_pixels=isolated,
    source_class_SHA256=sha(source),output_class_SHA256=sha(dest/(t+'_two_classes.png')),ownership_SHA256=sha(source_owner),
    RGB_SHA256=r['registered_RGB_SHA256'],RGB_relative_path=r['canonical_view_relative_path'])
   rows.append(row);src_hashes[str(source)]=sha(source)
  print('TWO_CLASS_RATES',pos,len(rr),flush=True)
 assert len(rows)==66 and sum(r['baseline']for r in rows)==18 and sum(r['is_adjacent']for r in rows)==43
 csvout(W/'V90_two_class_all_66_frames.csv',rows)
 # Retain planned missing cells as missing, without numeric imputation.
 for pos in sorted(set(r['position_id']for r in rows)):
  for t in ['T1','T2','T3','T4']:
   r=next((r for r in rows if r['position_id']==pos and r['period']==t),None)
   if r:full.append(dict(acquisition_status='observed',**r))
   else:
    blank={k:None for k in rows[0]};blank.update(position_id=pos,tube_id=tube_id(pos),material=pos.split('-')[0],period=t);full.append(dict(acquisition_status='not_acquired',**blank))
 csvout(W/'V90_two_class_all_72_planned_cells.csv',full)
 for name,kwargs in [
  ('Series_mean_adjacent',dict(regime='adjacent_only')),
  ('Series_mean_previous_observed',dict(regime='previous_observed')),
  ('Series_mean_tube_balanced',dict(regime='adjacent_only',tube_balanced=True)),
  ('Series_mean_excluding_coarse',dict(regime='adjacent_only',exclude_coarse=True)),
  ('Series_mean_assigned_support',dict(regime='adjacent_only',assigned=True))]:csvout(W/(name+'.csv'),summarize(rows,**kwargs))
 tube_rows=[]
 for tb in sorted(set(r['tube_id']for r in rows)):
  for regime in ['adjacent_only','previous_observed']:
   q=[r for r in rows if r['tube_id']==tb]
   for r in summarize(q,regime):
    if r['material']==tb.split('-')[0]:tube_rows.append(dict(tube_id=tb,**r))
 csvout(W/'Per_tube_two_class_means.csv',tube_rows)
 audit=dict(passed=True,actual_positions=18,actual_tubes=8,observed_images=66,planned_cells=72,missing_cells=6,baseline_frames=18,
  temporal_frames=48,real_adjacent_pairs=43,cross_gap_pairs=5,
  primary_rate_unit='Current skeleton-edge length fraction (%)',primary_mean='Unweighted per-position mean over real adjacent pairs, descriptive only',
  new_class_unchanged=True,existing_is_persistent_plus_extension=True,class_lengths_conserved=True,
  root_instance_maps_unchanged=True,SD_recomputed_after_grouping=True,baseline_rates_not_imputed=True,
  model_weights_or_annotations_changed=False,training_or_neural_inference_run=False,independent_accuracy_available=False,
  old_workbook_model_scope='Historical V35/V66; not used as the V90 rate source',source_prediction_SHA256=src_hashes)
 (W/'Two_class_checks.json').write_text(json.dumps(audit,indent=2));print('ALL_TWO_CLASS_RATES_COMPLETE',flush=True)
if __name__=='__main__':main()
