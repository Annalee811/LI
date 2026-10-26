"""Review every existing whole-root track without replacing native geometry.

Only endpoint-through continuation and one-sided lateral evidence are applied.
Unresolvable crossings retain their previous assignment and a review record.
"""
import collections,copy,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_whole_axis_v57_20261026';PREV=ROOT/'results/B3_root_continuity_review_v56_20261026'
def write(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n')
def unit(v):return v/max(np.linalg.norm(v),1e-9)
def tangent(p):
 p=np.asarray(p,float);i=np.arange(len(p));d=p[np.minimum(i+7,len(p)-1)]-p[np.maximum(i-7,0)];return d/np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-9)
class UF:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)
def main():
 sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
 raw=(PREV/'root_structure_draft.json').read_bytes();d=json.loads(raw);original=copy.deepcopy(d);st=d['root_structure_annotation'];assert require_active_position(st['position'])=='0353-2-B3';fs=st['periods'];hier={h['root_id']:copy.deepcopy(h) for h in st['root_track_hierarchy']};original_hier=copy.deepcopy(hier);(OUT/'source_v56.json').write_bytes(raw)
 teacher_points=[];teacher_t=[];teacher_r=[]
 for s in fs[0]['segments']:
  teacher_points.extend(s['points']);teacher_t.extend(tangent(s['points']));teacher_r.extend([s['root_id']]*len(s['points']))
 tp=np.asarray(teacher_points);tt=np.asarray(teacher_t);tr=np.asarray(teacher_r);tree=cKDTree(tp)
 changes=[];connections=[];issues=[];gaps=[];splits=[];parentaudit=[]
 def separate_family_axes(a,b):return a!=b and hier[a].get('family_root_id')==hier[b].get('family_root_id')
 def support(group):
  p=np.concatenate([s['points'] for s in group]);t=np.concatenate([tangent(s['points']) for s in group]);dist,ix=tree.query(p,k=8,distance_upper_bound=7);valid=ix<len(tp);ix=np.minimum(ix,len(tp)-1);alignment=np.abs(np.sum(t[:,None,:]*tt[ix],axis=2));score=np.exp(-dist**2/25)*alignment**8;score[~valid|(alignment<.9)]=0;best=score.argmax(axis=1);votes=collections.Counter()
  for i,j in enumerate(best):
   if score[i,j]>.25:votes[str(tr[ix[i,j]])]+=float(score[i,j])
  return votes
 for fi in [3,0,1,2]:
  f=fs[fi];segments=f['segments'];by={s['id']:s for s in segments};ix={s['id']:i for i,s in enumerate(segments)};uf=UF(len(segments));accepted=[];used=set()
  for n in f['nodes']:
   arms=[a for a in n['arms'] if a.get('direction')];candidates=[]
   for a,b in itertools.combinations(arms,2):
    sa,sb=by[a['segment_id']],by[b['segment_id']]
    if sa['id']==sb['id'] or separate_family_axes(sa['root_id'],sb['root_id']):continue
    pa,pb=np.asarray(sa['points']),np.asarray(sb['points']);da=min(np.linalg.norm(pa[0]-[n['x'],n['y']]),np.linalg.norm(pa[-1]-[n['x'],n['y']]));db=min(np.linalg.norm(pb[0]-[n['x'],n['y']]),np.linalg.norm(pb[-1]-[n['x'],n['y']]));cos=float(-np.dot(a['direction'],b['direction']))
    if cos>.94 and max(da,db)<=12:candidates.append((cos,a,b))
   # Distinct through-pairs at a crossing remain distinct; do not flood-fill.
   for cos,a,b in sorted(candidates,key=lambda x:-x[0]):
    others=[c[0] for c in candidates if (a['id'] in [c[1]['id'],c[2]['id']] or b['id'] in [c[1]['id'],c[2]['id']]) and {a['id'],b['id']}!={c[1]['id'],c[2]['id']}]
    if others and cos-max(others)<.06:continue
    sa,sb=by[a['segment_id']],by[b['segment_id']];pa,pb=np.asarray(sa['points']),np.asarray(sb['points']);center=np.array([n['x'],n['y']]);ea=(sa['id'],int(np.linalg.norm(pa[-1]-center)<np.linalg.norm(pa[0]-center)));eb=(sb['id'],int(np.linalg.norm(pb[-1]-center)<np.linalg.norm(pb[0]-center)))
    if ea in used or eb in used:continue
    used.update([ea,eb]);uf.join(ix[a['segment_id']],ix[b['segment_id']]);accepted.append(dict(period=f['period'],node_id=n['id'],point=[n['x'],n['y']],pair=[a['id'],b['id']],segments=[a['segment_id'],b['segment_id']],cosine=round(cos,4),basis='two terminal traces, unique opposed through-arms; crossing is not parentage',review_status='pending'))
  endpoints=[]
  for s in segments:
   p=np.asarray(s['points'])
   if len(p)<8:continue
   for e in [0,1]:endpoints.append((s,e,p[-1] if e else p[0],unit(p[-1]-p[-8] if e else p[0]-p[7])))
  et=cKDTree(np.asarray([e[2] for e in endpoints]));candidates=[]
  for i,j in et.query_pairs(7):
   a,b=endpoints[i],endpoints[j]
   if a[0]['id']==b[0]['id'] or separate_family_axes(a[0]['root_id'],b[0]['root_id']):continue
   dist=float(np.linalg.norm(a[2]-b[2]));cos=float(-np.dot(a[3],b[3]))
   if dist<.01 or cos<.975:continue
   if dist>2 and (np.dot(a[3],b[2]-a[2])/dist<.7 or np.dot(b[3],a[2]-b[2])/dist<.7):continue
   candidates.append((cos-dist*.01,a,b,dist,cos))
  endpoint_used=set(used)
  for score,a,b,dist,cos in sorted(candidates,key=lambda x:-x[0]):
   ea=(a[0]['id'],a[1]);eb=(b[0]['id'],b[1])
   if ea in endpoint_used or eb in endpoint_used:continue
   rivals=[c[0] for c in candidates if ((c[1][0]['id'],c[1][1]) in [ea,eb] or (c[2][0]['id'],c[2][1]) in [ea,eb]) and {c[1][0]['id'],c[2][0]['id']}!={a[0]['id'],b[0]['id']}]
   if rivals and score-max(rivals)<.08:continue
   uf.join(ix[a[0]['id']],ix[b[0]['id']]);endpoint_used.update([ea,eb]);gaps.append(dict(period=f['period'],segments=[a[0]['id'],b[0]['id']],distance=round(dist,3),cosine=round(cos,4),point=np.round((a[2]+b[2])/2,2).tolist(),review_status='pending'))
  grouped=collections.defaultdict(list)
  for i,s in enumerate(segments):grouped[uf.find(i)].append(s)
  for group in grouped.values():
   roots={s['root_id'] for s in group}
   if len(roots)<2:continue
   votes=support(group);current=collections.Counter()
   for s in group:current[s['root_id']]+=max(s['length_px'],1)
   # Name the route from the earliest clear match, with length fallback; never
   # shorten/recolor the actual mask or make a new ID at an extension boundary.
   winner=votes.most_common(1)[0][0] if votes and votes.most_common(1)[0][1]/sum(votes.values())>.72 else current.most_common(1)[0][0]
   if 'R0250' in roots and any(s['root_id']=='R0250' and s['length_px']>60 for s in group):winner='R0250'
   for s in group:
    if s['root_id']!=winner:
     # An established mother is not allowed to absorb a whole unrelated or
     # newly born lateral merely because one endpoint meets it. Existing-axis
     # reassignment needs sustained geometric support from T1.
     teacher=tp[tr==winner]
     if len(teacher)>=20 and s['length_px']>15:
      close=float(np.mean(cKDTree(teacher).query(s['points'])[0]<8));codes=s.get('semantic_code_counts',{});extension=codes.get('2',0)/max(1,sum(codes.values()))
      if close<.70 and extension<.60:
       issues.append(dict(period=f['period'],node_id=None,point=s['points'][len(s['points'])//2],segment_id=s['id'],root_id=s['root_id'],candidate=winner,reason='junction contact without sustained early-axis match; do not absorb lateral/unrelated root'));continue
     changes.append(dict(period=f['period'],segment_id=s['id'],point=s['points'][len(s['points'])//2],previous_root_id=s['root_id'],root_id=winner,basis='same through-route; earliest axis/continuous length support',review_status='pending'));s['root_id']=winner;s['review_status']='pending'
  for c in accepted:
   n=next(n for n in f['nodes'] if n['id']==c['node_id']);pair=c['pair'];n['connections']=[p for p in n['connections'] if not set(p)&set(pair)]+[pair]
  connections+=accepted
 # Newly visible one-sided branches sharing a mother's ID are split only with
 # T4 and another date supporting the same branch path. Unclear cases are queued.
 nextroot=max(int(r[1:]) for r in hier)+1
 t4=fs[3];by={s['id']:s for s in t4['segments']}
 for n in t4['nodes']:
  arms=n['arms']
  if len(arms)!=3 or any('direction' not in a for a in arms):continue
  ranked=sorted([(float(-np.dot(a['direction'],b['direction'])),i,j) for i,a in enumerate(arms) for j,b in enumerate(arms) if i<j],reverse=True);cos,i,j=ranked[0];k=3-i-j;ma,mb,side=[by[arms[x]['segment_id']] for x in [i,j,k]]
  if cos<.92:continue
  if cos-ranked[1][0]<.10:
   issues.append(dict(period='T4',node_id=n['id'],point=[n['x'],n['y']],reason='mother/side versus crossing has competing tangents; RGB review needed'));continue
  if ma['root_id']!=mb['root_id']:continue
  mother=ma['root_id'];child=side['root_id'];p=np.asarray(side['points']);center=np.array([n['x'],n['y']]);endpoint=min(np.linalg.norm(p[0]-center),np.linalg.norm(p[-1]-center));direction=np.asarray(arms[k]['direction']);proj=(p[np.linalg.norm(p-center,axis=1)<50]-center)@direction
  if endpoint>10 or np.count_nonzero(proj<-8)>max(3,.12*np.count_nonzero(proj>8)):continue
  if child==mother:
   if side['id'] in [ma['id'],mb['id']] or side['length_px']<30:continue
   peers=[];ptree=cKDTree(p);pt=tangent(p)
   for fi in [0,1,2]:
    for s in fs[fi]['segments']:
     if s['root_id']!=mother or s['length_px']<20:continue
     pp=np.asarray(s['points']);dist,idx=ptree.query(pp);align=np.abs(np.sum(tangent(pp)*pt[idx],axis=1));coverage=float(np.mean((dist<7)&(align>.9)))
     if coverage>.7:peers.append((fi,s,coverage))
   if not peers:issues.append(dict(period='T4',node_id=n['id'],point=center.tolist(),reason='possible same-ID lateral; no reliable earlier counterpart'));continue
   child='R'+str(nextroot).zfill(4);nextroot+=1;side['root_id']=child
   for fi,s,c in peers:s['root_id']=child
   first=min([3]+[fi for fi,s,c in peers]);hier[child]=dict(root_id=child,label=None,parent_root_id=mother,family_root_id=hier[mother]['family_root_id'],branch_kind='existing' if first==0 else 'new',review_status='pending',parentage_confirmed=False,first_seen_period='T'+str(first+1),observed_periods=[])
   splits.append(dict(root_id=child,mother_root_id=mother,period='T4',node_id=n['id'],point=center.tolist(),segments=[side['id']]+[s['id'] for fi,s,c in peers],previously_shared_mother_id=True,review_status='pending'))
  if child!=mother:
   original_parent=hier[child].get('parent_root_id');old=hier[child].get('label') or ''
   # Whole-track contact must be one-sided, including other fragments of child.
   local=np.concatenate([s['points'] for s in t4['segments'] if s['root_id']==child]);local=local[np.linalg.norm(local-center,axis=1)<50];projection=(local-center)@direction
   if np.count_nonzero(projection<-8)>max(4,.15*np.count_nonzero(projection>8)):continue
   parentaudit.append(dict(root_id=child,parent_root_id=mother,previous_parent=original_parent,period='T4',node_id=n['id'],point=center.tolist(),cosine=round(cos,4),review_status='pending'))
 # Rebuild the entire hierarchy consistently. Retain clear manually named
 # mothers and only change parentage with T4 one-sided attachment evidence.
 present=collections.defaultdict(list)
 for f in fs:
  for rid in {s['root_id'] for s in f['segments']}:present[rid].append(f['period'])
 for rid,h in list(hier.items()):
  p=h.get('parent_root_id');seen={rid}
  while p and p not in present and p not in seen:seen.add(p);p=hier[p].get('parent_root_id')
  h['parent_root_id']=p if p in present and p!=rid else None
 hier={r:h for r,h in hier.items() if r in present}
 for r,h in hier.items():
  first=min(present[r]);segments=[s for f in fs if f['period']==first for s in f['segments'] if s['root_id']==r];codes=collections.Counter()
  for s in segments:codes.update({int(k):v for k,v in s.get('semantic_code_counts',{}).items()})
  h['first_visibility_evidence_v57']=dict(period=first,new_class_fraction=round(codes[1]/max(1,sum(codes.values())),4),axis_length_px=round(sum(s['length_px'] for s in segments),3),first_visible_is_not_proven_birth=True)
  if h.get('parent_root_id') and first=='T1' and sum(s['length_px'] for s in segments)>=30:h['branch_kind']='existing'
 for e in parentaudit:
  r,p=e['root_id'],e['parent_root_id']
  if r not in hier or p not in hier or min(present[p])>min(present[r]):continue
  if original_hier.get(r,{}).get('review_status')=='modified' and not original_hier[r].get('parent_root_id'):continue
  corroboration=[]
  for f in fs[:3]:
   by={s['id']:s for s in f['segments']}
   for n in f['nodes']:
    if len(n['arms'])!=3 or np.linalg.norm(np.array([n['x'],n['y']])-e['point'])>12:continue
    mainarms=[a for a in n['arms'] if by[a['segment_id']]['root_id']==p];sidearms=[a for a in n['arms'] if by[a['segment_id']]['root_id']==r]
    if len(mainarms)==2 and len(sidearms)==1 and all('direction' in a for a in mainarms) and -np.dot(mainarms[0]['direction'],mainarms[1]['direction'])>.90:corroboration.append(dict(period=f['period'],node_id=n['id']))
  e['corroborating_dates']=corroboration;e['applied']=False
  if not corroboration and e['previous_parent']!=p:
   issues.append(dict(period='T4',node_id=e['node_id'],point=e['point'],root_id=r,reason='T4 attachment proposal lacks a matching earlier junction; parent unchanged pending review'));continue
  cur=p;seen={r}
  while cur and cur not in seen:seen.add(cur);cur=hier[cur].get('parent_root_id')
  if cur:continue
  hier[r]['parent_root_id']=p;e['applied']=True
  if e['previous_parent']!=p:hier[r]['branch_kind']='new' if min(present[r])!='T1' and hier[r]['first_visibility_evidence_v57']['new_class_fraction']>=.5 else 'existing'
 kids=collections.defaultdict(list)
 for rid,h in hier.items():kids[h.get('parent_root_id')].append(rid)
 reserved_N={h['label'] for r,h in hier.items() if r in kids[None] and h.get('label','').startswith('N-') and '-' not in h['label'][2:]};used=set();numeric=max(int(h['label']) for h in original_hier.values() if h.get('label','').isdigit())+1;newn=1
 for rid in sorted(kids[None],key=lambda r:(min(present[r]),hier[r].get('label') or 'ZZ'+r)):
  h=hier[rid];label=h.get('label');first=int(min(present[rid])[1:])-1;segments=[s for s in fs[first]['segments'] if s['root_id']==rid];classes=collections.Counter()
  for s in segments:classes.update({int(k):v for k,v in s.get('semantic_code_counts',{}).items()})
  new=first>0 and classes[1]/max(sum(classes.values()),1)>.6 and sum(s['length_px'] for s in segments)>50
  existing_N=bool(label and label.startswith('N-') and '-' not in label[2:])
  if new and not existing_N:
   pp=np.concatenate([s['points'] for s in segments]);earlier=np.concatenate([s['points'] for f in fs[:first] for s in f['segments']]);new=float(np.mean(cKDTree(earlier).query(pp)[0]<7))<.25
  if existing_N:pass
  elif new:
   while 'N-'+str(newn).zfill(4) in used or 'N-'+str(newn).zfill(4) in reserved_N:newn+=1
   label='N-'+str(newn).zfill(4);newn+=1
  elif not label or '-' in label or label in used:label=str(numeric).zfill(4);numeric+=1
  h.update(label=label,family_root_id=rid,branch_kind='independent');used.add(label)
 def visit(rid):
  h=hier[rid];occupied=set()
  for child in sorted(kids[rid],key=lambda c:(original_hier.get(c,{}).get('parent_root_id')!=rid,hier[c].get('label') or 'ZZ'+c)):
   c=hier[child];old=original_hier.get(child,{});kind=c.get('branch_kind','existing');kind='new' if kind=='new' else 'existing';suffix=(old.get('label') or '').split('-')[-1] if old.get('parent_root_id')==rid else '1';number=int(suffix.lstrip('N')) if suffix.lstrip('N').isdigit() else 1;prefix='N' if kind=='new' else ''
   while prefix+str(number) in occupied or h['label']+'-'+prefix+str(number) in used:number+=1
   occupied.add(prefix+str(number));c.update(label=h['label']+'-'+prefix+str(number),family_root_id=h['family_root_id'],branch_kind=kind);used.add(c['label']);visit(child)
 for rid in kids[None]:visit(rid)
 for rid,h in hier.items():h.update(first_seen_period=min(present[rid]),observed_periods=sorted(present[rid]),review_status='pending',parentage_confirmed=False)
 for rid,h in hier.items():
  oldlabel=original_hier.get(rid,{}).get('label')
  if oldlabel and oldlabel!=h['label']:h['previous_display_labels_v57']=list(dict.fromkeys(h.get('previous_display_labels_v57',[])+[oldlabel]));h['label_revision_reason_v57']='consistent lifetime mother/side hierarchy; T1-visible baseline sides use existing-side numbering'
 for f in fs:
  by={s['id']:s for s in f['segments']}
  for s in f['segments']:
   h=hier[s['root_id']];s.update(display_label=h['label'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'])
  for n in f['nodes']:
   keep=[]
   for pair in n['connections']:
    roots=[by[next(a for a in n['arms'] if a['id']==x)['segment_id']]['root_id'] for x in pair]
    if roots[0]==roots[1]:keep.append(pair)
    else:issues.append(dict(period=f['period'],node_id=n['id'],point=[n['x'],n['y']],reason='old connection conflicts with whole-root ownership; review needed'))
   n['connections']=keep
 st['root_track_hierarchy']=list(hier.values());st['root_tracks']=[dict(root_id=r,display_label=h['label'],first_seen_period=h['first_seen_period'],observed_periods=h['observed_periods'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'],review_status='pending',biological_identity_confirmed=False) for r,h in hier.items()];st['cross_period_links']=[]
 for rid,h in hier.items():
  obs=[(f,[s for s in f['segments'] if s['root_id']==rid]) for f in fs];obs=[(f,g) for f,g in obs if g]
  for (a,ga),(b,gb) in zip(obs,obs[1:]):st['cross_period_links'].append(dict(id='L-'+rid+'-'+b['period'],previous_period=a['period'],current_period=b['period'],root_id=rid,display_label=h['label'],previous_segment=ga[0]['id'],current_segment=gb[0]['id'],previous_segments=[s['id'] for s in ga],current_segments=[s['id'] for s in gb],kind='same_root_candidate',review_status='pending'))
 final_connections=[];final_gaps=[]
 for c in connections:
  f=next(f for f in fs if f['period']==c['period']);by={s['id']:s for s in f['segments']};node=next(n for n in f['nodes'] if n['id']==c['node_id'])
  if by[c['segments'][0]]['root_id']==by[c['segments'][1]]['root_id'] and c['pair'] in node['connections']:final_connections.append(c)
 for g in gaps:
  f=next(f for f in fs if f['period']==g['period']);by={s['id']:s for s in f['segments']}
  if by[g['segments'][0]]['root_id']==by[g['segments'][1]]['root_id']:final_gaps.append(g)
  else:issues.append(dict(period=g['period'],point=g['point'],node_id=None,reason='aligned endpoint gap conflicts with lateral ownership; requires RGB review'))
 connections=final_connections;gaps=final_gaps
 audit=dict(source_SHA256=hashlib.sha256(raw).hexdigest(),all_periods_all_segments_revisited=True,changes=changes,parent_proposals=parentaudit,lateral_splits=splits,uncertain_junctions=issues,accepted_through_connections=connections,aligned_gap_connections=gaps,all_new_assignments_require_human_review=True,method='preserve native traces; unique terminal through-arms and aligned endpoint gaps; earliest identity anchors; one-sided T4 lateral evidence')
 st['whole_axis_revision_v57']=audit;st['status']='proposal_under_review';st['human_reviewed']=False
 assert d['images']==original['images'];assert len({h['label'] for h in hier.values()})==len(hier)
 for f,old in zip(fs,original['root_structure_annotation']['periods']):assert f['owner_map_rle']==old['owner_map_rle'];assert [s['points'] for s in f['segments']]==[s['points'] for s in old['segments']]
 # The known 0001 main route is a regression guard across the entire field.
 main=next(h['root_id'] for h in hier.values() if h['label']=='0001')
 for f,old in zip(fs,original['root_structure_annotation']['periods']):
  by={s['id']:s for s in f['segments']}
  for s in old['segments']:
   if s['root_id']==main and s['length_px']>60:assert by[s['id']]['root_id']==main
 write('root_structure_draft.json',d);write('revision_audit.json',audit);stats=dict(passed=True,all_segments_revisited=True,all_native_pixel_layers_unchanged=True,all_native_trace_coordinates_unchanged=True,owner_map_exactly_unchanged=True,unique_lifetime_labels=True,same_id_on_accepted_continuations=True,root0001_guard_passed=True,checked_segments=sum(len(f['segments']) for f in fs),identity_changes=len(changes),accepted_through_connections=len(connections),aligned_gaps=len(gaps),T4_parent_candidates=len(parentaudit),lateral_splits=len(splits),uncertain_junctions=len(issues),independent_N_candidates=sum(h['label'].startswith('N-') and not h.get('parent_root_id') for h in hier.values()))
 write('draft_verification.json',stats);print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
