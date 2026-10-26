import hashlib,json
from pathlib import Path
OUT=Path('/workspace/LI/results/B3_root_temporal_review_v59_20261026')
def main():
 d=json.loads((OUT/'root_structure_draft.json').read_text());src=json.loads((OUT/'source_progress_14.json').read_text());st=d['root_structure_annotation'];ss=src['root_structure_annotation'];h={x['root_id']:x for x in st['root_track_hierarchy']};sh={x['root_id']:x for x in ss['root_track_hierarchy']};a=st['temporal_revision_v59'];f=a['T1_changes_explicitly_requested_in_screenshot'];exceptions={(x['period'],x['segment_id']) for x in f['reassigned_segments']};parent_exceptions={x['root_id'] for x in f['parent_changes']};renamed={x['root_id'] for x in f['renames']};checks={}
 checks['T1_pixel_layers_exact']=d['images'][0]==src['images'][0]
 checks['T1_geometry_nodes_and_ownership_exact']={k:v for k,v in st['periods'][0].items() if k!='segments'}=={k:v for k,v in ss['periods'][0].items() if k!='segments'} and [(q['id'],q['points']) for q in st['periods'][0]['segments']]==[(q['id'],q['points']) for q in ss['periods'][0]['segments']]
 checks['T1_other_root_assignments_unchanged']=all(s['root_id']==o['root_id'] for s,o in zip(st['periods'][0]['segments'],ss['periods'][0]['segments']) if ('T1',s['id']) not in exceptions)
 t1ids={q['root_id'] for q in ss['periods'][0]['segments']};checks['T1_other_display_labels_unchanged']=all(h[r]['label']==sh[r]['label'] for r in t1ids-renamed-parent_exceptions)
 checks['all_other_parent_edges_preserved']=all(h[r].get('parent_root_id')==x.get('parent_root_id') for r,x in sh.items() if r not in parent_exceptions)
 checks['manual_assignments_unchanged_except_explicit_screenshot_family']=all(x['root_id']==y['root_id'] for ff,sf in zip(st['periods'],ss['periods']) for x,y in zip(ff['segments'],sf['segments']) if y.get('review_status') in ['modified','confirmed'] and (ff['period'],x['id']) not in exceptions)
 checks['all_trace_coordinates_and_owner_maps_exact']=all(ff['owner_map_rle']==sf['owner_map_rle'] and [(q['id'],q['points']) for q in ff['segments']]==[(q['id'],q['points']) for q in sf['segments']] for ff,sf in zip(st['periods'],ss['periods']))
 checks['all_non_temporal_pixel_layers_exact']=all(all(v==src['images'][j]['layers'][k] for k,v in im['layers'].items() if k!='temporal') for j,im in enumerate(d['images']))
 checks['unique_global_labels']=len(h)==len({x['label'] for x in h.values()})
 checks['upper_and_lower_axes_have_distinct_fixed_ids_all_periods']=all(next(q for q in ff['segments'] if q['id']==top)['root_id']==f['upper_root_id'] and next(q for q in ff['segments'] if q['id']==lower)['root_id']==f['lower_root_id'] for ff,(top,lower) in zip(st['periods'],[('T1-S0005','T1-S0199'),('T2-S0006','T2-S0727'),('T3-S0007','T3-S0713'),('T4-S0006','T4-S0293')]))
 checks['lower_existing_root_has_no_false_N_prefix']=not h[f['lower_root_id']]['label'].startswith('N-') and h[f['lower_root_id']]['first_seen_period']=='T1'
 checks['lower_child_parent_is_local_not_remote_upper']=h['R0143']['parent_root_id']==f['lower_root_id'] and h['R0143']['family_root_id']!=f['upper_root_id']
 checks['N_candidates_retain_id_and_label_when_observed']=all(all(s['display_label']==h[x['root_id']]['label'] for ff in st['periods'] for s in ff['segments'] if s['root_id']==x['root_id']) for x in a['new_independent_numbering'])
 checks['lifetime_observation_states_are_actual']=all(len(t['states'])==4 and all(q['observed']==any(s['root_id']==t['root_id'] for s in st['periods'][j]['segments']) for j,q in enumerate(t['states'])) for t in st['root_lifetime_registry_v58'])
 checks['parent_cycles_absent']=True
 for r in h:
  seen=set();q=r
  while q:
   if q in seen or q not in h:checks['parent_cycles_absent']=False;break
   seen.add(q);q=h[q].get('parent_root_id')
 checks['original_upload_hash_intact']=hashlib.sha256((OUT/'source_progress_14.json').read_bytes()).hexdigest()==a['source_SHA256'];print(json.dumps(checks,indent=2));assert all(checks.values())
 (OUT/'data_integrity_checks.json').write_text(json.dumps(dict(passed=True,checks=checks,final_json_SHA256=hashlib.sha256((OUT/'root_structure_draft.json').read_bytes()).hexdigest()),indent=2)+'\n')
if __name__=='__main__':main()
