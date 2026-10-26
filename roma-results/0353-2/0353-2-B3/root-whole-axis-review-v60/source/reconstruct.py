"""Whole-axis paths first, dated correspondence second, identity labels last.

No foreground or native trace coordinates are synthesized. Native RGB and
through-tangents are used to reconnect existing pieces. Dated routes, rather
than old IDs or connected foreground blobs, define correspondence candidates.
"""
import collections,copy,hashlib,itertools,json,sys,os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_whole_trace_v60_20261026'
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_parentage_v58'))
from refine import tangent,geometry,root_groups
class UF:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)
def write(name,v):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n');t.replace(p)
def unit(v):return v/max(np.linalg.norm(v),1e-9)
def covered(a,b,radius=6):
 p=a['p'];dist,ix=b['tree'].query(p,k=min(4,len(b['p'])))
 if dist.ndim==1:dist=dist[:,None];ix=ix[:,None]
 angle=np.abs(np.sum(a['t'][:,None,:]*b['t'][ix],axis=2));valid=(dist<radius)&(angle>.91);score=np.exp(-dist**2/32)*angle**8;score[~valid]=0
 best=score.argmax(1);dd=dist[np.arange(len(p)),best];aa=angle[np.arange(len(p)),best];ok=valid.any(1)
 return float(ok.mean()),int(ok.sum()),float(np.median(dd[ok])) if ok.any() else 999,float(np.median(aa[ok])) if ok.any() else 0,float(score.max(1).mean())
def rgb_bridge(rgb,a,b):
 p=np.linspace(a,b,max(3,int(np.linalg.norm(a-b))*2));v=unit(b-a);normal=np.array([-v[1],v[0]]);samples=[]
 for offset in [0,-4,4]:
  q=np.rint(p+normal*offset).astype(int);q[:,0]=q[:,0].clip(0,rgb.shape[1]-1);q[:,1]=q[:,1].clip(0,rgb.shape[0]-1);samples.append(rgb[q[:,1],q[:,0]].mean(1))
 return float(np.median(samples[0]-(samples[1]+samples[2])/2)/255)
def dated_paths(f,rgb):
 ss=f['segments'];by={s['id']:i for i,s in enumerate(ss)};uf=UF(len(ss));ends=[];eindex={};candidates={}
 for i,s in enumerate(ss):
  p=np.asarray(s['points'])
  if len(p)<5 or s['length_px']<5:continue
  for e in [0,1]:
   ep=p[-1] if e else p[0];vec=unit(ep-(p[max(0,len(p)-10)] if e else p[min(9,len(p)-1)]));eindex[i,e]=len(ends);ends.append((i,e,ep,vec))
 def offer(a,b,cos,basis,node=None):
  if a==b or ends[a][0]==ends[b][0]:return
  a,b=sorted([a,b]);dist=float(np.linalg.norm(ends[a][2]-ends[b][2]));score=cos-dist*.004+(0.015 if node else 0)
  if (a,b) not in candidates or score>candidates[a,b]['score']:candidates[a,b]=dict(a=a,b=b,score=score,alignment=cos,distance=dist,basis=basis,node_id=node)
 # Node directions see through a pixel-owned crossing; each endpoint can be
 # used once. Parent/side status and old colors never veto physical continuity.
 for n in f['nodes']:
  arms=n['arms'];center=np.array([n['x'],n['y']]);options=[]
  for a,b in itertools.combinations(arms,2):
   if a['segment_id']==b['segment_id'] or 'direction' not in a or 'direction' not in b:continue
   ia,ib=by[a['segment_id']],by[b['segment_id']];pa=np.asarray(ss[ia]['points']);pb=np.asarray(ss[ib]['points']);ea=int(np.linalg.norm(pa[-1]-center)<np.linalg.norm(pa[0]-center));eb=int(np.linalg.norm(pb[-1]-center)<np.linalg.norm(pb[0]-center))
   if (ia,ea) not in eindex or (ib,eb) not in eindex:continue
   x,y=eindex[ia,ea],eindex[ib,eb];dist=max(np.linalg.norm(ends[x][2]-center),np.linalg.norm(ends[y][2]-center));cos=float(-np.dot(a['direction'],b['direction']));actual=float(-np.dot(ends[x][3],ends[y][3]))
   if dist<=12 and cos>.93 and actual>.87:offer(x,y,min(cos,actual+.03),'unique opposed arms at a crossing',n['id'])
 if ends:
  et=cKDTree([e[2] for e in ends])
  for i,j in et.query_pairs(16):
   a,b=ends[i],ends[j]
   if a[0]==b[0]:continue
   delta=b[2]-a[2];dist=float(np.linalg.norm(delta));cos=float(-a[3]@b[3])
   if cos<(.93 if dist<=3 else .965):continue
   if dist>3:
    forward=min(float(a[3]@unit(delta)),float(b[3]@unit(-delta)))
    if forward<.72:continue
   if dist>8 and (cos<.985 or rgb_bridge(rgb,a[2],b[2])<.012):continue
   offer(i,j,cos,'aligned endpoints with native RGB bridge' if dist>8 else 'aligned neighboring endpoints')
 # Mutual unique best pairs prevent a junction from becoming a branching
 # connected component and swallowing a lateral or the other crossing root.
 options=collections.defaultdict(list)
 for c in candidates.values():options[c['a']].append(c);options[c['b']].append(c)
 for v in options.values():v.sort(key=lambda c:-c['score'])
 accepted=[];uncertain=[];used=set()
 for c in sorted(candidates.values(),key=lambda c:-c['score']):
  a,b=c['a'],c['b'];competing=[x['score'] for q in [a,b] for x in options[q] if x is not c]
  if a in used or b in used:continue
  if options[a][0] is not c or options[b][0] is not c or competing and c['score']-max(competing)<(.0 if os.environ.get('ROOT_PREVIEW60')=='1' else .045):
   if ends[a][0]!=ends[b][0]:uncertain.append(c)
   continue
  uf.join(ends[a][0],ends[b][0]);used.update([a,b]);accepted.append(c)
 groups=collections.defaultdict(list)
 for i,s in enumerate(ss):groups[uf.find(i)].append(s)
 paths=[]
 for v in groups.values():
  g=geometry(v);g.update(segments=v,length=sum(s['length_px'] for s in v),bbox=np.array([g['p'].min(0),g['p'].max(0)]));paths.append(g)
 def record(c):return dict(period=f['period'],segments=[ss[ends[c[k]][0]]['id'] for k in ['a','b']],point=((ends[c['a']][2]+ends[c['b']][2])/2).tolist(),alignment=round(c['alignment'],4),endpoint_distance=round(c['distance'],3),basis=c['basis'],node_id=c['node_id'])
 return paths,[record(c) for c in accepted],[record(c) for c in uncertain]
def main():
 raw=(OUT/'source_v59.json').read_bytes();source=json.loads(raw);d=copy.deepcopy(source);st=d['root_structure_annotation'];frames=st['periods'];hier={h['root_id']:h for h in st['root_track_hierarchy']};oldhier=copy.deepcopy(hier)
 sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
 assert require_active_position(st['position'])=='0353-2-B3'
 routes=[];connections=[];uncertain=[];dated=[]
 for fi,f in enumerate(frames):
  rgb=np.asarray(Image.open(ROOT/'results/B3_temporal_unet_aligned_v21_20261021/data/upper'/f'T{fi+1}_RGB.png').convert('RGB'));paths,links,issues=dated_paths(f,rgb);connections.extend(links);uncertain.extend(issues)
  for path in paths:path['fi']=fi;path['index']=len(routes);routes.append(path)
  dated.append(paths);print('dated paths',f['period'],len(paths),'joined boundaries',len(links),flush=True)
 uf=UF(len(routes));members={i:{i} for i in range(len(routes))};temporal=[];rejected=[]
 def join_routes(a,b,e):
  ra,rb=uf.find(a),uf.find(b)
  if ra==rb:return True
  # Parallel competing routes with overlapping support in one frame must not
  # collapse into one track. Disjoint portions of the same prior axis may join.
  for ia in members[ra]:
   for ib in members[rb]:
    x,y=routes[ia],routes[ib]
    if x['fi']!=y['fi'] or min(len(x['p']),len(y['p']))<12:continue
    if (x['bbox'][1]+4<y['bbox'][0]).any() or (y['bbox'][1]+4<x['bbox'][0]).any():continue
    overlap=covered(x,y,4)[0] if len(x['p'])<len(y['p']) else covered(y,x,4)[0]
    if overlap>.28:rejected.append(dict(**e,reason='same-date competing routes overlap; not automatically merged'));return False
  combined=members.pop(ra)|members.pop(rb);uf.join(ra,rb);members[uf.find(ra)]=combined;return True
 for gap in [1,2,3]:
  plans=[]
  for fi in range(4-gap):
   for a in dated[fi]:
    if a['length']<12:continue
    for b in dated[fi+gap]:
     if b['length']<12 or (a['bbox'][1]+8<b['bbox'][0]).any() or (b['bbox'][1]+8<a['bbox'][0]).any():continue
     ab=covered(a,b);ba=covered(b,a);small=ab if len(a['p'])<=len(b['p']) else ba
     if small[0]<.78 or small[1]<18 or small[2]>4.5 or small[3]<.95:continue
     score=small[0]+.20*max(ab[0],ba[0])+.10*min(ab[0],ba[0])+.1*small[3]-.025*small[2]-.035*(gap-1)
     e=dict(previous_period='T'+str(fi+1),period='T'+str(fi+gap+1),previous_route=a['index'],route=b['index'],previous_segments=[s['id'] for s in a['segments']],segments=[s['id'] for s in b['segments']],coverage_previous=round(ab[0],4),coverage_current=round(ba[0],4),matched_points=small[1],median_distance=round(small[2],3),tangent_alignment=round(small[3],4),score=score)
     plans.append((score,a['index'],b['index'],e))
  # Competing routes with effectively identical evidence are explicitly held.
  bypair=collections.defaultdict(list)
  for score,a,b,e in plans:bypair[a,routes[b]['fi']].append((score,b));bypair[b,routes[a]['fi']].append((score,a))
  for score,a,b,e in sorted(plans,reverse=True,key=lambda x:x[0]):
   ambiguous=False
   for q,other in [(a,b),(b,a)]:
    rivals=sorted(bypair[q,routes[other]['fi']],reverse=True)
    if len(rivals)>1 and rivals[0][0]-rivals[1][0]<.075:
     x,y=routes[rivals[0][1]],routes[rivals[1][1]]
     if min(len(x['p']),len(y['p']))>=12:
      cv=covered(x,y)[0] if len(x['p'])<len(y['p']) else covered(y,x)[0]
      if cv>.28:ambiguous=True
   if ambiguous:rejected.append(dict(**e,reason='near-equal competing dated axes'));continue
   if join_routes(a,b,e):temporal.append(e)
  print('date gap checked',gap,'accepted',len(temporal),flush=True)
 tracks=collections.defaultdict(list)
 for p in routes:tracks[uf.find(p['index'])].append(p)
 write('candidate_paths.json',dict(candidate_only=True,source_SHA256=hashlib.sha256(raw).hexdigest(),tracks=[dict(key=key,routes=[dict(period='T'+str(p['fi']+1),segments=[s['id'] for s in p['segments']],length_px=round(p['length'],3)) for p in ps]) for key,ps in tracks.items()],connections=connections,temporal=temporal,ambiguous=rejected))
 def depth(r):
  seen=set();n=0
  while hier[r].get('parent_root_id') and r not in seen:seen.add(r);r=hier[r]['parent_root_id'];n+=1
  return n
 rankings={};contributors={}
 for key,ps in tracks.items():
  votes=collections.defaultdict(float);lengths=collections.defaultdict(lambda:collections.defaultdict(float))
  for p in ps:
   for s in p['segments']:lengths[s['root_id']][p['fi']]+=max(s['length_px'],.1)
  contributors[key]=lengths;earliest=min(p['fi'] for p in ps)
  for r,dd in lengths.items():
   first=min(dd);v=dd[first]*4+sum(dd.values())*.25+60*(4-first)
   if first==earliest:v+=80
   if dd.get(0,0)>30:v+=120/(depth(r)+1)
   # Established main-axis IDs stay ahead of a falsely split descendant when
   # both actually contribute substantial native T1 geometry to this route.
   if dd.get(0,0)>30 and depth(r)==0:v+=250
   if r=='R0250' and dd.get(0,0)>60:v+=10000
   votes[r]=v
  rankings[key]=sorted(votes.items(),key=lambda q:-q[1])
 assigned={};used=set();unresolved_id_components=[];nextroot=max(int(r[1:]) for r in hier)+1;nextlabel=max([int(h['label']) for h in hier.values() if h['label'].isdigit()]+[0])+1
 for key in sorted(tracks,key=lambda k:-rankings[k][0][1]):
  bestscore=rankings[key][0][1];candidates=[r for r,v in rankings[key] if r not in used and v>=bestscore*.8]
  if candidates:r=candidates[0]
  else:
   # Lack of geometric linkage is NOT evidence for a different root or birth.
   # Retain the prior identity for unresolved disconnected portions instead of
   # generating a new ID for every missing/occluded dated component.
   r=rankings[key][0][0];unresolved_id_components.append(dict(root_id=r,routes=[p['index'] for p in tracks[key]],reason='earlier identity retained across an unresolved geometric gap; no new numbered root fabricated'))
  assigned[key]=r;used.add(r)
 changes=[];routeaudit=[]
 for key,ps in tracks.items():
  r=assigned[key]
  for p in ps:
   old_ids={s['root_id'] for s in p['segments']}
   routeaudit.append(dict(period='T'+str(p['fi']+1),route=p['index'],root_id=r,previous_root_ids=sorted(old_ids),segments=[s['id'] for s in p['segments']],length_px=round(p['length'],3)))
   for s in p['segments']:
    old=s['root_id']
    if old!=r:changes.append(dict(period='T'+str(p['fi']+1),segment_id=s['id'],previous_root_id=old,root_id=r,point=s['points'][len(s['points'])//2],reason='same complete through-axis and dated route correspondence'));s['root_id']=r;s['review_status']='pending'
 # Provisional pieces of one old root may be nonoverlapping date fragments
 # without sufficient geometric overlap. Do not call lack of evidence a new
 # birth: retain aliases and expose the unresolved tracks for manual review.
 present=collections.defaultdict(list);grouped=[root_groups(f) for f in frames]
 for fi,g in enumerate(grouped):
  for r in g:present[r].append(fi)
 live={r:copy.deepcopy(hier[r]) for r in present};parent_reviews=[]
 # Recover which canonical root actually contains an old local mother segment.
 origin=collections.defaultdict(set)
 for key,dd in contributors.items():
  for oldr in dd:origin[oldr].add(assigned[key])
 for r,h in live.items():
  original_parent=h.get('parent_root_id');parents=origin.get(original_parent,set())-{r};ev=[]
  for parent in parents:
   for fi in present[r]:
    if parent not in grouped[fi]:continue
    endpoints=np.array([p for s in grouped[fi][r] for p in [s['points'][0],s['points'][-1]]]);distance=float(geometry(grouped[fi][parent])['tree'].query(endpoints)[0].min())
    if distance<=12:ev.append(dict(period='T'+str(fi+1),parent_root_id=parent,distance_px=round(distance,3)))
  best=collections.Counter(e['parent_root_id'] for e in ev);ranked=best.most_common();newparent=ranked[0][0] if ranked and (len(ranked)==1 or ranked[0][1]>ranked[1][1]) else None
  h['parent_root_id']=newparent
  if newparent!=original_parent:parent_reviews.append(dict(root_id=r,previous_parent=original_parent,parent_root_id=newparent,evidence=ev,review_status='pending',reason='mother identity follows the local whole-axis segment, not a retired or remote number'));h['parentage_confirmed']=False
 # Break an unsupported cycle explicitly; never disguise it with a relabel.
 for r in live:
  cur=r;seen=set()
  while cur and cur in live:
   if cur in seen:live[r]['parent_root_id']=None;parent_reviews.append(dict(root_id=r,reason='cycle after axis merge withheld',parent_root_id=None));break
   seen.add(cur);cur=live[cur].get('parent_root_id')
 children=collections.defaultdict(list)
 for r,h in live.items():children[h.get('parent_root_id')].append(r)
 usedlabels=set();renames=[]
 for r in children[None]:
  h=live[r];label=h['label']
  if '-' in label and not (label.startswith('N-') and label[2:].isdigit()) or label in usedlabels:
   label=str(nextlabel).zfill(4);nextlabel+=1
  if label!=h['label']:renames.append(dict(root_id=r,previous_label=h['label'],label=label));h['label']=label
  usedlabels.add(label);h['family_root_id']=r;h['branch_kind']='independent'
 def visit(r,family):
  h=live[r];h['family_root_id']=family;reserved=set()
  for c in children[r]:
   q=live[c];old=q['label'];part=old.split('-')[-1];new=q.get('branch_kind')=='new';digits=part.lstrip('N');n=int(digits) if digits.isdigit() else 1
   while ('N' if new else '')+str(n) in reserved or h['label']+'-'+('N' if new else '')+str(n) in usedlabels:n+=1
   suffix=('N' if new else '')+str(n);reserved.add(suffix);label=h['label']+'-'+suffix;usedlabels.add(label);q['label']=label
   if old!=label:renames.append(dict(root_id=c,previous_label=old,label=label))
   visit(c,family)
 for r in children[None]:visit(r,r)
 lifetime=[];gaps=[]
 for r,h in live.items():
  dates=present[r];h['first_seen_period']='T'+str(dates[0]+1);h['observed_periods']=['T'+str(j+1) for j in dates];h['birth_role_v58']=('new_lateral' if h.get('branch_kind')=='new' else 'existing_lateral') if h.get('parent_root_id') else 'baseline_independent' if dates[0]==0 else h.get('birth_role_v58','unresolved_side_vs_independent');states=[]
  for j in range(4):
   status='before_first_observation' if j<dates[0] else 'observed_same_identity' if j in dates else 'observation_gap_identity_retained' if max(dates)>j else 'not_reliably_matched_identity_retained';states.append(dict(period='T'+str(j+1),status=status,root_id=r,label=h['label'],parent_root_id=h.get('parent_root_id'),observed=j in dates))
   if j>=dates[0] and j not in dates:gaps.append(dict(root_id=r,display_label=h['label'],period='T'+str(j+1),status=status,point=grouped[dates[0]][r][0]['points'][0]))
  lifetime.append(dict(root_id=r,label=h['label'],parent_root_id=h.get('parent_root_id'),birth_role=h.get('birth_role_v58'),states=states,identity_retained_through_T4=True,visibility_is_not_identity=True))
 for f in frames:
  for s in f['segments']:
   h=live[s['root_id']];s.update(display_label=h['label'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'])
 st['root_track_hierarchy']=list(live.values());st['root_lifetime_registry_v58']=lifetime;st['root_tracks']=[dict(root_id=r,display_label=h['label'],first_seen_period=h['first_seen_period'],observed_periods=h['observed_periods'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'],review_status='pending',biological_identity_confirmed=False) for r,h in live.items()];st['cross_period_links']=[]
 for r,dates in present.items():
  for a,b in zip(dates,dates[1:]):
   ga,gb=grouped[a][r],grouped[b][r];st['cross_period_links'].append(dict(id='L-'+r+'-T'+str(b+1),previous_period='T'+str(a+1),current_period='T'+str(b+1),root_id=r,display_label=live[r]['label'],previous_segment=ga[0]['id'],current_segment=gb[0]['id'],previous_segments=[s['id'] for s in ga],current_segments=[s['id'] for s in gb],kind='same_axis_correspondence_candidate',review_status='pending'))
 audit=dict(source_SHA256=hashlib.sha256(raw).hexdigest(),native_pixels_and_geometry_fixed=True,method='whole-axis through-pair graph, geometric dated route correspondences, then stable identity naming',priority='T2 T3 T4 physical continuity; no identity-only gap fill',within_period_connections=connections,unresolved_endpoint_choices=uncertain,dated_axis_matches=temporal,rejected_ambiguous_matches=rejected,segment_identity_changes=changes,dated_routes=routeaudit,parent_reviews=parent_reviews,renames=renames,unresolved_prior_identity_components=unresolved_id_components,visibility_gaps=gaps,training_started=False)
 st['whole_trace_revision_v60']=audit;st['status']='whole_axis_proposal_under_review';st['human_reviewed']=False
 assert d['images']==source['images'];assert len({h['label'] for h in live.values()})==len(live)
 for f,sf in zip(frames,source['root_structure_annotation']['periods']):assert f['owner_map_rle']==sf['owner_map_rle'] and [(s['id'],s['points']) for s in f['segments']]==[(s['id'],s['points']) for s in sf['segments']]
 write('root_structure_draft.json',d);write('revision_audit.json',audit)
 stats=dict(passed=True,source_SHA256=audit['source_SHA256'],all_pixel_layers_exact=True,all_trace_coordinates_and_single_owner_maps_exact=True,within_period_connections=len(connections),dated_axis_matches=len(temporal),old_live_IDs=len(oldhier),whole_axis_track_candidates=len(live),changed_segments=len(changes),changes_by_period=dict(collections.Counter(e['period'] for e in changes)),unresolved_endpoint_choices=len(uncertain),ambiguities_withheld=len(rejected),visibility_gap_records=len(gaps),final_json_SHA256=hashlib.sha256((OUT/'root_structure_draft.json').read_bytes()).hexdigest());write('draft_verification.json',stats);print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
