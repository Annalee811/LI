"""Check final uploaded13 lineage and preserve explicit user node edits."""
import hashlib,json
from pathlib import Path
OUT=Path('/workspace/LI/results/B3_root_parentage_v58_20261026');PREV=Path('/workspace/LI/results/B3_root_whole_axis_v57_20261026')
def main():
 d=json.loads((OUT/'root_structure_draft.json').read_text());source=json.loads((OUT/'source_progress_13.json').read_text());old=json.loads((PREV/'root_structure_draft.json').read_text());st=d['root_structure_annotation'];audit=st['parentage_revision_v58'];conflicts=[]
 for f,src,prev in zip(st['periods'],source['root_structure_annotation']['periods'],old['root_structure_annotation']['periods']):
  by={s['id']:s for s in f['segments']};oldnodes={n['id']:n for n in prev['nodes']};srcnodes={n['id']:n for n in src['nodes']}
  assert f['owner_map_rle']==src['owner_map_rle'];assert [(s['id'],s['points']) for s in f['segments']]==[(s['id'],s['points']) for s in src['segments']]
  for n in f['nodes']:
   protected=n.get('review_status') in ['modified','confirmed'] or n['id'] not in oldnodes or srcnodes[n['id']].get('connections')!=oldnodes[n['id']].get('connections');keep=[]
   for pair in n['connections']:
    roots=[by[next(a for a in n['arms'] if a['id']==arm)['segment_id']]['root_id'] for arm in pair]
    if roots[0]==roots[1]:keep.append(pair)
    else:
     conflicts.append(dict(period=f['period'],node_id=n['id'],point=[n['x'],n['y']],root_ids=roots,pair=pair,manual_connection_retained=protected,reason='connection conflicts with current whole-root identity; do not merge roots just to satisfy an old pair'))
     if protected:keep.append(pair)
   n['connections']=keep
 assert d['images']==source['images'];h={x['root_id']:x for x in st['root_track_hierarchy']};assert len({x['label'] for x in h.values()})==len(h)
 for r,x in h.items():
  seen=set();cur=r
  while cur:assert cur not in seen and cur in h;seen.add(cur);cur=h[cur].get('parent_root_id')
  actual=[f['period'] for f in st['periods'] if any(s['root_id']==r for s in f['segments'])];assert actual==x['observed_periods']
  if x.get('parent_root_id'):
   prefix=h[x['parent_root_id']]['label']+'-';assert x['label'].startswith(prefix) and '-' not in x['label'][len(prefix):]
 for t in st['root_lifetime_registry_v58']:
  assert len(t['states'])==4 and t['states'][3]['root_id']==t['root_id']
  assert all(s['root_id']==t['root_id'] and s['label']==t['label'] and s['parent_root_id']==t['parent_root_id'] for s in t['states'])
  assert all(s['observed']==any(q['root_id']==t['root_id'] for q in st['periods'][i]['segments']) for i,s in enumerate(t['states']))
 audit['junction_identity_conflicts']=conflicts
 (OUT/'root_structure_draft.json').write_text(json.dumps(d,ensure_ascii=False,separators=(',',':'))+'\n');(OUT/'revision_audit.json').write_text(json.dumps(audit,ensure_ascii=False,separators=(',',':'))+'\n');v=json.loads((OUT/'draft_verification.json').read_text());v.update(hierarchy_and_true_observations_verified=True,old_automatic_conflicting_connections_removed=sum(not e['manual_connection_retained'] for e in conflicts),manual_connection_conflicts_retained=sum(e['manual_connection_retained'] for e in conflicts),final_json_SHA256=hashlib.sha256((OUT/'root_structure_draft.json').read_bytes()).hexdigest());(OUT/'draft_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');print(json.dumps(v,indent=2))
if __name__=='__main__':main()
