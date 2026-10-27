import json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
R=Path('/workspace/LI');O=R/'results/B3_left_region_review_v63_20261027';D=json.loads((O/'source_progress_16.json').read_text());S=D['root_structure_annotation'];H={h['root_id']:h for h in S['root_track_hierarchy']}
f=S['periods'][0]
for index in [65,85,124,138,139,164,166,170,224,243,268,275,282,315,334]:
 s=next(s for s in f['segments'] if s['index']==index);q=np.array(s['points']);tree=cKDTree(q)
 print('\nBASE',index,'length',round(s['length_px']),s['points'][0],s['points'][-1])
 # endpoints' closest already-defined root and native location
 for p in [q[0],q[-1]]:
  nearest=[]
  for v in f['segments']:
   if H[v['root_id']].get('annotation_status_v60')=='unassigned':continue
   ds=np.linalg.norm(np.array(v['points'])-p,axis=1);k=ds.argmin();nearest.append((round(ds[k],1),H[v['root_id']]['label'],v['points'][k]))
  print('  formal-near',p.tolist(),sorted(nearest)[:2])
 for peer in S['periods'][1:]:
  found=[]
  for v in peer['segments']:
   if H[v['root_id']].get('annotation_status_v60')!='unassigned':continue
   pp=np.array(v['points']);dd=tree.query(pp)[0];rev=cKDTree(pp).query(q)[0];overlap=np.sum(dd<10);frac=overlap/max(1,len(pp));rfrac=np.mean(rev<10)
   if overlap>=3 and (frac>.45 or rfrac>.45):found.append((v['index'],round(v['length_px']),round(frac,2),round(rfrac,2),v['points'][0],v['points'][-1]))
  print(' ',peer['period'],found)
