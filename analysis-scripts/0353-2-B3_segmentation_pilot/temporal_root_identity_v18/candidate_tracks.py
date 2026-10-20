"""Expose candidate root identities and previous-date links without claiming ground truth."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[]
for m in c.MODELS:
 previous={};next_id=1
 for t in c.PERIODS:
  axis=c.load_axes(a.run/'axis_cache'/m/f'{t}.npz');records=json.loads((a.run/m/'four_period_prediction'/f'{t}_root_axes.json').read_text()) if t!='T1' else None;current={};mapping=np.zeros(len(axis['paths'])+1,np.uint16)
  for root,path in enumerate(axis['paths'],1):
   rec=records[root-1] if records else {};parent=rec.get('matched_previous_axis',0);new=rec.get('new_nodes',0)>0;linked=not new and parent in previous and rec.get('previous_support_fraction',0)>=.50 and rec.get('dominant_previous_fraction',0)>=.70
   if linked:track=previous[parent];status='Candidate inherited previous root'
   else:track=next_id;next_id+=1;status='Newly observed candidate root' if new else ('T1 candidate baseline' if t=='T1' else 'Identity unresolved; separate provisional ID')
   current[root]=track;mapping[root]=track;rows.append(dict(model=m,period=t,root_axis_id=root,candidate_track_id=track,matched_previous_axis=parent,status=status,newly_observed=new,previous_visibility=rec.get('root_previous_visibility_fraction',rec.get('previous_support_fraction',None)),previous_match_dominance=rec.get('dominant_previous_fraction',None),unresolved_crossing=rec.get('unresolved_crossing',False),identity_independently_verified=False))
  Image.fromarray(mapping[axis['axis_map']]).save(a.run/m/'four_period_prediction'/f'{t}_candidate_track_ids_uint16.png');previous=current
c.csvout(a.run/'tables/candidate_root_tracks.csv',rows);c.js(a.run/'tables/candidate_tracks_protocol.json',dict(candidate_IDs_only=True,inheritance_requires_predicted_previous_root_support=True,NEW_roots_never_inherit_old_track_id=True,unmatched_root_identity_not_called_death=True,no_independent_instance_labels=True));print('CANDIDATE_TRACK_TABLE_READY',len(rows),flush=True)
