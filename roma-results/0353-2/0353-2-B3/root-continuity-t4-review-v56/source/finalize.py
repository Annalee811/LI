"""Check live references and remove contradictory automatic graph connections."""
from pathlib import Path
import collections,hashlib,json
import numpy as np
from scipy.spatial import cKDTree
OUT=Path('/workspace/LI/results/B3_root_continuity_review_v56_20261026')
def main():
 d=json.loads((OUT/'root_structure_draft.json').read_text());source=json.loads((OUT/'source_progress_12.json').read_text());st=d['root_structure_annotation'];audit=st['continuity_revision_v56'];changes=list(audit.get('connection_conflicts_removed',[]));manual=list(audit.get('manual_connection_conflicts',[]));repairs=list(audit.get('endpoint_continuation_repairs',[]))
 first=collections.defaultdict(list)
 for s in st['periods'][0]['segments']:first[s['root_id']].extend(s['points'])
 trees={r:cKDTree(p) for r,p in first.items() if len(p)>20}
 # Fix short misnumbered islands bracketed at BOTH ends by the same other
 # axis. This catches the 0001 fragments inside 0001-N5; no single-ended or
 # perpendicular crossing is enough, and no T1 trajectory is reassigned.
 for f in st['periods'][1:]:
  for s in f['segments']:
   if s.get('review_status')=='confirmed' or len(s['points'])<8 or s['length_px']>70 or s['root_id'] not in trees:continue
   p=np.asarray(s['points']);
   if np.median(trees[s['root_id']].query(p)[0])<25:continue
   winners=[]
   for end in [0,-1]:
    pp=p[end];v=pp-p[7 if end==0 else -8];v=v/max(np.linalg.norm(v),1e-9);candidates=[]
    for q in f['segments']:
     if q['id']==s['id'] or len(q['points'])<8:continue
     qp=np.asarray(q['points'])
     for k in [0,-1]:
      distance=float(np.linalg.norm(qp[k]-pp));vv=qp[k]-qp[7 if k==0 else -8];vv=vv/max(np.linalg.norm(vv),1e-9);alignment=float(-np.dot(v,vv))
      if distance<=6 and alignment>.97:candidates.append((alignment*math_exp(-distance/15),q['root_id'],q['id'],distance,alignment))
    candidates.sort(reverse=True)
    if not candidates:break
    other=[x for x in candidates[1:] if x[1]!=candidates[0][1]]
    if other and candidates[0][0]-other[0][0]<.1:break
    winners.append(candidates[0])
   target=winners[0][1] if len(winners)==2 and winners[0][1]==winners[1][1]!=s['root_id'] else None;peer_evidence=[]
   if not target and len(winners)==2:
    # Beside a crossing the endpoint tangent can briefly prefer the other
    # crossing root. Two separate dates covering >=85% of the actual short
    # trace within 5px resolve this; a one-point junction match cannot.
    proposed=[]
    for candidate in {x[1] for x in winners if x[1]!=s['root_id']}:
     evidence=[]
     for peer in st['periods']:
      if peer is f:continue
      pp=[q['points'] for q in peer['segments'] if q['root_id']==candidate]
      if not pp:continue
      coverage=float(np.mean(cKDTree(np.concatenate(pp)).query(p)[0]<5))
      if coverage>=.85:evidence.append(dict(period=peer['period'],coverage=round(coverage,4)))
     if len(evidence)>=2:proposed.append((candidate,evidence))
    if len(proposed)==1:target,peer_evidence=proposed[0]
   if target:
    repairs.append(dict(period=f['period'],segment_id=s['id'],previous_root_id=s['root_id'],root_id=target,point=s['points'][len(s['points'])//2],neighbors=[x[2] for x in winners],endpoint_distances=[round(x[3],3) for x in winners],tangent_agreements=[round(x[4],4) for x in winners],multi_date_evidence=peer_evidence,review_status='pending',basis='short identity island; two aligned same-root endpoints' if not peer_evidence else 'crossing-adjacent short island: one aligned endpoint plus two whole-trace matches in separate dates'))
    s['root_id']=target;s['review_status']='pending';s['endpoint_repair_v56']=repairs[-1]
 for f in st['periods']:
  by={s['id']:s for s in f['segments']}
  for n in f['nodes']:
   arms={a['id']:a for a in n['arms']};good=[]
   for pair in n['connections']:
    roots=[by[arms[a]['segment_id']]['root_id'] for a in pair]
    if roots[0]!=roots[1]:
     entry=dict(period=f['period'],node_id=n['id'],point=[n['x'],n['y']],pair=pair,root_ids=roots,reason='stored automatic connection joins different current root identities',review_status='pending')
     if n['review_status'] in ['modified','confirmed']:manual.append(entry);good.append(pair)
     else:changes.append(entry)
    else:good.append(pair)
   n['connections']=good
 st['continuity_revision_v56']['connection_conflicts_removed']=changes;st['continuity_revision_v56']['manual_connection_conflicts']=manual;st['continuity_revision_v56']['endpoint_continuation_repairs']=repairs
 assert d['images']==source['images'];hier={h['root_id']:h for h in st['root_track_hierarchy']};assert len({h['label'] for h in hier.values()})==len(hier)
 for rid,h in hier.items():
  h['observed_periods']=[f['period'] for f in st['periods'] if any(s['root_id']==rid for s in f['segments'])];h['first_seen_period']=h['observed_periods'][0]
 for f in st['periods']:
  for s in f['segments']:
   h=hier[s['root_id']];s.update(display_label=h['label'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'])
 st['root_tracks']=[dict(root_id=rid,display_label=h['label'],first_seen_period=h['first_seen_period'],observed_periods=h['observed_periods'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'],review_status=h['review_status'],biological_identity_confirmed=False) for rid,h in hier.items()]
 st['cross_period_links']=[]
 for rid,h in hier.items():
  obs=[(f,[s for s in f['segments'] if s['root_id']==rid]) for f in st['periods']];obs=[(f,g) for f,g in obs if g]
  for (a,ga),(b,gb) in zip(obs,obs[1:]):st['cross_period_links'].append(dict(id='L-'+rid+'-'+b['period'],previous_period=a['period'],current_period=b['period'],root_id=rid,display_label=h['label'],previous_segment=ga[0]['id'],current_segment=gb[0]['id'],previous_segments=[s['id'] for s in ga],current_segments=[s['id'] for s in gb],kind='same_root_candidate',review_status='pending'))
 for f in st['periods']:
  ids={s['id'] for s in f['segments']};indices={s['index'] for s in f['segments']};rle=f['owner_map_rle'];assert sum(rle[1::2])==f['width']*f['height'];assert set(rle[::2])<=(indices|{0})
  for n in f['nodes']:
   assert not n.get('owner_segment_id') or n['owner_segment_id'] in ids
   assert all(a['segment_id'] in ids for a in n['arms'])
 for rid,h in hier.items():
  seen=set();cur=rid
  while cur:
   assert cur not in seen and cur in hier;seen.add(cur);cur=hier[cur].get('parent_root_id')
  parent=h.get('parent_root_id')
  if parent:
   assert h['label'].startswith(hier[parent]['label']+'-');suffix=h['label'][len(hier[parent]['label'])+1:];assert '-' not in suffix
  observed=[f['period'] for f in st['periods'] if any(s['root_id']==rid for s in f['segments'])];assert observed==h['observed_periods']
 audit=st['continuity_revision_v56'];(OUT/'root_structure_draft.json').write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n');(OUT/'revision_audit.json').write_text(json.dumps(audit,ensure_ascii=False,separators=(',',':'))+'\n')
 path=OUT/'draft_verification.json';v=json.loads(path.read_text());v.pop('other_protected_assignments_unchanged',None);v.update(removed_contradictory_automatic_connections=len(changes),manual_connection_conflicts=len(manual),endpoint_continuation_repairs=len(repairs),unique_owner_per_pixel=True,hierarchy_acyclic_and_unique=True,all_observation_dates_rebuilt=True,family0001_observations={h['label']:h['observed_periods'] for h in hier.values() if h['label']=='0001' or h['label'].startswith('0001-')},final_json_SHA256=hashlib.sha256((OUT/'root_structure_draft.json').read_bytes()).hexdigest());path.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in v.items() if k!='family0001_observations'},indent=2))
def math_exp(value):return float(np.exp(value))
if __name__=='__main__':main()
