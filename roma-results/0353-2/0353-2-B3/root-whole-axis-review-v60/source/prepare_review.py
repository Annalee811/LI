"""Publish conservative annotation, never unverified automatic root numbers."""
import collections,copy,hashlib,json
from pathlib import Path
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_whole_trace_v60_20261026'
def main():
 raw=(OUT/'source_v59.json').read_bytes();d=json.loads(raw);source=copy.deepcopy(d);st=d['root_structure_annotation'];paths=json.loads((OUT/'candidate_paths.json').read_text());oldh={h['root_id']:h for h in st['root_track_hierarchy']};frames=st['periods'];by={(f['period'],s['id']):s for f in frames for s in f['segments']};claimed={};h=[];known={}
 known['R0151']={key for key,s in by.items() if s['root_id']=='R0151'}
 horizontal=[['T1-S0319','T1-S0323'],['T2-S0473','T2-S0735','T2-S0736'],['T3-S0471','T3-S0480','T3-S0729','T3-S0730','T3-S0731','T3-S0732'],['T4-S0478','T4-S0488','T4-S0784','T4-S0785','T4-S0786','T4-S0787']]
 known['R0148']={(f'T{i+1}',sid) for i,row in enumerate(horizontal) for sid in row}
 for rid,keys in known.items():
  entry=copy.deepcopy(oldh[rid]);entry.update(annotation_status_v60='reviewed_visible_axis',parent_root_id=None,family_root_id=rid,branch_kind='independent',review_status='assistant_visual_review',reviewed_by='assistant_visual_review',confirmation_basis_v60='native RGB four-period visible-axis review; no claim of unseen root length or biological parentage');h.append(entry)
  for key in keys:assert key in by;claimed[key]=rid
 count=0;proposal_audit=[]
 for t in paths['tracks']:
  keys={(r['period'],sid) for r in t['routes'] for sid in r['segments']}-claimed.keys()
  if not keys:continue
  count+=1;rid='R'+str(20000+count);label='U-'+str(count).zfill(4);previous=collections.Counter(by[key]['root_id'] for key in keys)
  entry=dict(root_id=rid,label=label,parent_root_id=None,family_root_id=rid,branch_kind='independent',review_status='pending',parentage_confirmed=False,annotation_status_v60='unassigned',hidden_technical_label=True,candidate_path_only=True,previous_root_ids_v60=list(previous),previous_display_labels_v60=[oldh[r]['label'] for r in previous],birth_role_v58='unresolved_side_vs_independent');h.append(entry)
  for key in keys:claimed[key]=rid
  proposal_audit.append(dict(internal_root_id=rid,technical_label=label,biological_number_assigned=False,routes=t['routes'],previous_display_labels=entry['previous_display_labels_v60']))
 assert len(claimed)==len(by)
 hm={x['root_id']:x for x in h};observed=collections.defaultdict(set)
 for fi,f in enumerate(frames):
  for s in f['segments']:
   rid=claimed[f['period'],s['id']];q=hm[rid];s.update(root_id=rid,display_label=q['label'],parent_root_id=None,family_root_id=rid,branch_kind='independent',review_status='confirmed' if q['annotation_status_v60']=='reviewed_visible_axis' else 'pending');observed[rid].add(fi)
 life=[]
 for rid,q in hm.items():
  dates=observed[rid];q['first_seen_period']='T'+str(min(dates)+1);q['observed_periods']=['T'+str(j+1) for j in sorted(dates)];q['birth_role_v58']='baseline_independent' if min(dates)==0 and q['annotation_status_v60']!='unassigned' else 'unresolved_side_vs_independent'
  states=[dict(period='T'+str(j+1),root_id=rid,label=q['label'],parent_root_id=None,observed=j in dates,status='before_first_observation' if j<min(dates) else 'observed_same_identity' if j in dates else 'observation_gap_identity_retained' if j<max(dates) else 'not_reliably_matched_identity_retained') for j in range(4)]
  life.append(dict(root_id=rid,label=q['label'],parent_root_id=None,birth_role=q['birth_role_v58'],states=states,identity_retained_through_T4=True,visibility_is_not_identity=True))
 st['root_track_hierarchy']=h;st['root_lifetime_registry_v58']=life;st['root_tracks']=[dict(root_id=q['root_id'],display_label=q['label'],parent_root_id=None,family_root_id=q['root_id'],branch_kind='independent',first_seen_period=q['first_seen_period'],observed_periods=q['observed_periods'],review_status=q['review_status'],biological_identity_confirmed=False,annotation_status_v60=q['annotation_status_v60']) for q in h];st['cross_period_links']=[]
 for rid,q in hm.items():
  dates=sorted(observed[rid])
  for a,b in zip(dates,dates[1:]):
   sa=[s['id'] for s in frames[a]['segments'] if s['root_id']==rid];sb=[s['id'] for s in frames[b]['segments'] if s['root_id']==rid];st['cross_period_links'].append(dict(id=f'L-{rid}-T{b+1}',previous_period='T'+str(a+1),current_period='T'+str(b+1),root_id=rid,display_label=q['label'],previous_segment=sa[0],current_segment=sb[0],previous_segments=sa,current_segments=sb,kind='same_visible_axis_reviewed' if rid in known else 'unverified_candidate_only',review_status='confirmed' if rid in known else 'pending'))
 st['confidence_review_v60']=dict(source_SHA256=hashlib.sha256(raw).hexdigest(),known_root_ids=list(known),known_labels=[hm[r]['label'] for r in known],formal_numbers_assigned_only_to_reviewed_visible_axes=True,unassigned_tracks=count,unassigned_tracks_are_not_biological_root_counts=True,uncertain_paths_hidden_until_selected=True,technical_U_labels_not_displayed_as_root_numbers=True,new_numbers_only_after_user_definition=True,unknown_candidate_paths=proposal_audit,unseen_pixels_not_synthesized=True,all_native_pixels_and_geometry_unchanged=True,training_started=False)
 st['human_reviewed']=False;st['status']='conservative_visible_axis_review_and_user_definition';st['numbering_schema']='v60: known T1 IDs; N-xxxx new independent; mother-Nx new lateral; U-xxxx internal unassigned path handle (not a biological root number)'
 assert d['images']==source['images']
 for f,sf in zip(frames,source['root_structure_annotation']['periods']):assert f['owner_map_rle']==sf['owner_map_rle'] and [(s['id'],s['points']) for s in f['segments']]==[(s['id'],s['points']) for s in sf['segments']]
 text=json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n';(OUT/'review_project.json').write_text(text)
 report=dict(passed=True,source_SHA256=hashlib.sha256(raw).hexdigest(),all_native_pixel_layers_exact=True,all_trace_coordinates_and_owners_exact=True,formal_reviewed_labels=[hm[r]['label'] for r in known],unassigned_internal_candidates=count,no_automatic_new_biological_numbers=True,formal_T1_labels_within_0312=all(1<=int(hm[r]['label'])<=312 for r in known),review_project_SHA256=hashlib.sha256(text.encode()).hexdigest());(OUT/'review_data_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
