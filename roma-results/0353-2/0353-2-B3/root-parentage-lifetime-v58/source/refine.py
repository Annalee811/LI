"""Parentage-first, durable identities on the uploaded13 annotation.

No missing observation is fabricated. A visibility gap retains the same identity
and is never evidence for a second birth. Root pixels and native traces stay fixed.
"""
import collections,copy,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_parentage_v58_20261026';PREV=ROOT/'results/B3_root_whole_axis_v57_20261026'
def write(n,v):
 tmp=OUT/(n+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n');tmp.replace(OUT/n)
def tangent(p):
 p=np.asarray(p,float);i=np.arange(len(p));v=p[np.minimum(i+7,len(p)-1)]-p[np.maximum(i-7,0)];return v/np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-9)
def root_groups(f):
 g=collections.defaultdict(list)
 for s in f['segments']:g[s['root_id']].append(s)
 return g
def geometry(g):
 p=np.concatenate([s['points'] for s in g]);return dict(p=p,t=np.concatenate([tangent(s['points']) for s in g]),tree=cKDTree(p))
def similarity(s,geom):
 p=np.asarray(s['points']);dist,idx=geom['tree'].query(p);alignment=np.abs(np.sum(tangent(p)*geom['t'][idx],axis=1));valid=(dist<6)&(alignment>.92);coverage=float(valid.mean());median=float(np.median(dist));return coverage,median,float(np.median(alignment)),float(np.mean(np.exp(-dist**2/18)*alignment**8))
def main():
 sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
 raw=(OUT/'source_progress_13.json').read_bytes();d=json.loads(raw);original=copy.deepcopy(d);st=d['root_structure_annotation'];assert require_active_position(st['position'])=='0353-2-B3';frames=st['periods'];hier={h['root_id']:h for h in st['root_track_hierarchy']};old=json.loads((PREV/'root_structure_draft.json').read_text())['root_structure_annotation'];oh={h['root_id']:h for h in old['root_track_hierarchy']}
 protected=set();protected_h=set()
 for h in hier.values():
  oldh=oh.get(h['root_id'],{})
  if h.get('label')!=oldh.get('label') or h.get('parent_root_id')!=oldh.get('parent_root_id') or h.get('parentage_confirmed'):protected_h.add(h['root_id'])
 for f,o in zip(frames,old['periods']):
  by={s['id']:s for s in o['segments']}
  for s in f['segments']:
   v=by.get(s['id'])
   if not v or s['root_id']!=v['root_id'] or s['points']!=v['points'] or s.get('review_status')=='confirmed':protected.add((f['period'],s['id']))
 for f in frames:
  for s in f['segments']:
   if (f['period'],s['id']) in protected:protected_h.add(s['root_id'])
 merges=[];uncertain=[];growth=[];distinct_laterals=[]
 def decode(a):return np.repeat(np.asarray(a[::2],np.int32),a[1::2])
 native_temporal=[decode(im['layers']['temporal']) for im in d['images']]
 # Elongation keeps the existing root identity, including an already named
 # lateral. Only terminal-to-terminal continuation with extension-class support
 # is eligible; a newly appearing lateral has a different birth role.
 for ti in [1,2,3]:
  f=frames[ti];prior_ids={s['root_id'] for oldf in frames[:ti] for s in oldf['segments']};temporal=native_temporal[ti]
  grown_keys=set()
  for iteration in range(2):
   groups=root_groups(f);ends=[]
   for r,ss in groups.items():
    if r not in prior_ids:continue
    persistent_core=sum(int(np.count_nonzero(np.isin(temporal[np.asarray(q['points'])[:,1]*f['width']+np.asarray(q['points'])[:,0]],[3,4]))) for q in ss)
    if persistent_core<25:continue
    for s in ss:
     p=np.asarray(s['points'])
     if len(p)<8:continue
     for end in [0,1]:
      point=p[-1] if end else p[0];direction=(point-p[-8]) if end else (point-p[7]);direction=direction/max(np.linalg.norm(direction),1e-9)
      if any(q['id']!=s['id'] and cKDTree(q['points']).query(point)[0]<6 for q in ss):continue
      ends.append((r,s,end,point,direction))
   if not ends:continue
   treeends=cKDTree([x[3] for x in ends]);plans=[]
   for s in f['segments']:
    if (f['period'],s['id']) in protected or s['root_id'] in protected_h or s['id'] in grown_keys or len(s['points'])<8:continue
    p=np.asarray(s['points']);values=temporal[p[:,1]*f['width']+p[:,0]];extension=float(np.mean(values==2))
    if extension<.60:continue
    candidates=[]
    for end in [0,1]:
     point=p[-1] if end else p[0];direction=point-p[-8] if end else point-p[7];direction=direction/max(np.linalg.norm(direction),1e-9)
     for ix in treeends.query_ball_point(point,7):
      r,body,be,bp,bd=ends[ix]
      if r==s['root_id']:continue
      distance=float(np.linalg.norm(point-bp));alignment=float(-np.dot(direction,bd))
      if alignment>.96:candidates.append((alignment-distance*.01,r,body['id'],distance,alignment))
    candidates.sort(reverse=True)
    if not candidates or len(candidates)>1 and candidates[0][1]!=candidates[1][1] and candidates[0][0]-candidates[1][0]<.10:continue
    score,r,body,distance,alignment=candidates[0];plans.append((s,r,body,extension,distance,alignment))
   for s,r,body,extension,distance,alignment in plans:
    previous=s['root_id'];s['root_id']=r;s['review_status']='pending';grown_keys.add(s['id']);growth.append(dict(period=f['period'],segment_id=s['id'],body_segment=body,previous_root_id=previous,root_id=r,extension_class_fraction=round(extension,4),endpoint_distance=round(distance,3),tangent_alignment=round(alignment,4),point=s['points'][0],basis='elongating terminal segment continues existing identity; no new root number'))
 # Detect a different failure: two physically separate laterals accidentally
 # sharing one ID. Distinct attachment sites on the SAME mother distinguish
 # this from elongation. Reviewed/edited parts are never moved.
 nextroot=max(int(r[1:]) for r in hier)+1;f4=frames[3]
 for r in sorted(protected_h):
  if r not in hier or not hier[r].get('parent_root_id'):continue
  ss=[s for s in f4['segments'] if s['root_id']==r];anchor=[s for s in ss if ('T4',s['id']) in protected];other=[s for s in ss if ('T4',s['id']) not in protected];mother=hier[r]['parent_root_id'];mp=[s['points'] for s in f4['segments'] if s['root_id']==mother]
  if not anchor or not other or not mp:continue
  mg=cKDTree(np.concatenate(mp));ag=geometry(anchor);ae=np.asarray([p for s in anchor for p in [s['points'][0],s['points'][-1]]]);ad=mg.query(ae)[0];ap=ae[int(ad.argmin())]
  if ad.min()>12:continue
  for s in other:
   if s['length_px']<35:continue
   p=np.asarray(s['points']);dd=ag['tree'].query(p)[0]
   if np.quantile(dd,.10)<12:continue
   ep=np.asarray([p[0],p[-1]]);distance=mg.query(ep)[0];origin=ep[int(distance.argmin())]
   if distance.min()>8 or np.linalg.norm(origin-ap)<25:continue
   matches=[]
   geom=geometry([s])
   for fi,ff in enumerate(frames):
    for q in ff['segments']:
     if q['root_id']!=r or (ff['period'],q['id']) in protected:continue
     coverage,dist,angle,score=similarity(q,geom)
     reverse=similarity(s,geometry([q]))
     # A later visible fragment can be shorter under partial occlusion. Test
     # both directions so an already present T1 root is not reborn at T2.
     same_curve=coverage>.82 and dist<3.5 and angle>.94
     same_curve|=coverage>.60 and reverse[0]>.90 and reverse[1]<3.5 and reverse[2]>.94
     if same_curve:matches.append((fi,q))
   if len({fi for fi,q in matches})<2:continue
   # A segment that leaves the terminal of the mother in the SAME direction
   # is elongation, not a second lateral. Test the whole local parent axis:
   # for elongation it lies behind the segment and has no through-arm ahead.
   direction=(p[min(10,len(p)-1)]-p[0]) if np.linalg.norm(p[0]-origin)<np.linalg.norm(p[-1]-origin) else (p[-min(11,len(p))]-p[-1]);direction=direction/max(np.linalg.norm(direction),1e-9);parent_points=np.concatenate(mp);local=parent_points[np.linalg.norm(parent_points-origin,axis=1)<40];projection=(local-origin)@direction;perpendicular=np.abs((local-origin)@np.array([-direction[1],direction[0]]));behind=int(np.count_nonzero(projection<-8));ahead=int(np.count_nonzero(projection>8));terminal_continuation=behind>=8 and ahead<=max(3,.12*behind) and float(np.median(perpendicular))<6
   if terminal_continuation:
    for fi,q in matches:
     previous=q['root_id'];q['root_id']=mother;q['review_status']='pending';growth.append(dict(period=frames[fi]['period'],segment_id=q['id'],previous_root_id=previous,root_id=mother,point=q['points'][0],basis='same-direction continuation of the mother terminal; no parent through-arm ahead; not a new lateral',parent_points_behind=behind,parent_points_ahead=ahead,temporal_class_review_needed=True))
    continue
   newr='R'+str(nextroot).zfill(4);nextroot+=1;first=min(fi for fi,q in matches)
   for fi,q in matches:q['root_id']=newr;q['review_status']='pending'
   retained_label=None;naming_note=None
   if first==0:
    remaining=[(fi,q) for fi,ff in enumerate(frames) for q in ff['segments'] if q['root_id']==r]
    if remaining and min(fi for fi,q in remaining)>0:
     fi=min(fi for fi,q in remaining);pp=np.concatenate([q['points'] for j,q in remaining if j==fi]);native=native_temporal[fi];new_fraction=float(np.mean(native[pp[:,1]*frames[fi]['width']+pp[:,0]]==1))
     if new_fraction>.6:retained_label=hier[r]['label'];hier[r]['label']=None;hier[r]['branch_kind']='new';naming_note='T1-existing lateral retains its original display number across all periods; the different manually edited later-born lateral receives its own N-side number'
   hier[newr]=dict(root_id=newr,label=retained_label,parent_root_id=mother,family_root_id=hier[mother]['family_root_id'],branch_kind='existing' if first==0 else 'new',review_status='pending',parentage_confirmed=False)
   distinct_laterals.append(dict(root_id=newr,previous_shared_root_id=r,parent_root_id=mother,origin=origin.tolist(),other_origin=ap.tolist(),origin_separation_px=round(float(np.linalg.norm(origin-ap)),3),observed_periods=['T'+str(fi+1) for fi in sorted({fi for fi,q in matches})],segments=[q['id'] for fi,q in matches],naming_note=naming_note,basis='separate lateral axes at separate stable mother attachment points; not a newly numbered extension'))
 # Only recover genuinely missing identities; do not replace an established
 # matched track with a nearby parallel root. Latest edited segments are locked.
 for ti in [1,2,3]:
  target=frames[ti];tg=root_groups(target);prior=[root_groups(f) for f in frames[:ti]];present=set(tg);seeds=collections.defaultdict(list)
  for fi,g in enumerate(prior):
   for r,ss in g.items():
    if r not in present and sum(s['length_px'] for s in ss)>=35:seeds[r].append((fi,geometry(ss)))
  for s in sorted(target['segments'],key=lambda s:-s['length_px']):
   if (target['period'],s['id']) in protected or s['length_px']<25 or len(s['points'])<20:continue
   p=np.asarray(s['points']);rank=[];current=s['root_id'];current_prior=[geometry(g[current]) for g in prior if current in g];current_support=max([similarity(s,g)[0] for g in current_prior]+[0])
   for r,gg in seeds.items():
    checks=[]
    for fi,g in gg:
     if (g['p'].max(axis=0)+7<p.min(axis=0)).any() or (g['p'].min(axis=0)-7>p.max(axis=0)).any():continue
     coverage,dist,angle,score=similarity(s,g)
     if coverage>=.82 and dist<=3.5 and angle>=.94:checks.append(dict(period='T'+str(fi+1),coverage=round(coverage,4),median_distance=round(dist,3),tangent_alignment=round(angle,4),score=score))
    if not checks:continue
    strongest=max(e['score'] for e in checks);eligible=len(checks)>=2 or r in protected_h
    if eligible and current_support<.45:rank.append((strongest,r,checks))
    elif strongest>.6:uncertain.append(dict(period=target['period'],segment_id=s['id'],point=s['points'][len(s['points'])//2],root_id=r,current_root_id=current,reason='missing-root continuation candidate lacks independent date support or current identity also fits',evidence=checks))
   rank.sort(reverse=True,key=lambda x:x[0])
   if not rank or len(rank)>1 and rank[0][0]-rank[1][0]<.16:continue
   score,r,checks=rank[0];s['root_id']=r;s['review_status']='pending';merges.append(dict(period=target['period'],segment_id=s['id'],point=s['points'][len(s['points'])//2],previous_root_id=current,root_id=r,evidence=checks,basis='same axis in earlier dates; retained identity, not a new birth'))
 # Rebuild dated geometry after recoveries, before any parentage decision.
 grouped=[root_groups(f) for f in frames];present=collections.defaultdict(list)
 for fi,g in enumerate(grouped):
  for r in g:present[r].append(fi)
 # T4 attachment evidence has priority, with corroborating dates and genuine
 # one-sided termination of the ENTIRE child, not a random crossing point.
 evidence=collections.defaultdict(list);junction_review=[]
 for fi in [3,2,1,0]:
  f=frames[fi];by={s['id']:s for s in f['segments']}
  for n in f['nodes']:
   a=n['arms']
   if len(a)!=3 or any('direction' not in x for x in a):continue
   pairs=sorted([(float(-np.dot(x['direction'],y['direction'])),i,j) for i,x in enumerate(a) for j,y in enumerate(a) if i<j],reverse=True);cos,i,j=pairs[0];k=3-i-j
   pa,pb,child=[by[a[x]['segment_id']]['root_id'] for x in [i,j,k]]
   if pa!=pb or child==pa or cos<.90:continue
   center=np.asarray([n['x'],n['y']]);direction=np.asarray(a[k]['direction']);sp=np.asarray(by[a[k]['segment_id']]['points']);ed=float(min(np.linalg.norm(sp[0]-center),np.linalg.norm(sp[-1]-center)));allp=np.concatenate([s['points'] for s in grouped[fi][child]]);local=allp[np.linalg.norm(allp-center,axis=1)<55];project=(local-center)@direction;back=int((project<-9).sum());forward=int((project>9).sum())
   if ed>10 or back>max(3,.10*forward):continue
   if present[pa][0]>present[child][0]:continue
   margin=cos-pairs[1][0];score=.55*cos+.20*max(0,1-ed/10)+.15*max(0,min(1,margin/.25))+.10*(fi==3)
   if margin<.08:
    junction_review.append(dict(period=f['period'],node_id=n['id'],point=center.tolist(),root_id=child,reason='ambiguous main/side directions; do not infer parent from intersection'));continue
   evidence[child,pa].append(dict(period=f['period'],node_id=n['id'],point=center.tolist(),main_pair=[a[i]['id'],a[j]['id']],lateral_arm=a[k]['id'],cosine=round(cos,4),endpoint_distance=round(ed,3),opposite_child_points=back,score=round(score,4)))
 def depth(r):
  seen=set();n=0
  while r in hier and hier[r].get('parent_root_id'):
   if r in seen:return 999
   seen.add(r);r=hier[r]['parent_root_id'];n+=1
  return n
 parent_changes=[];parent_reviews=[]
 for child in sorted({c for c,p in evidence}):
  candidates=[]
  for (c,p),ev in evidence.items():
   if c!=child:continue
   t4=any(e['period']=='T4' for e in ev);dates=len({e['period'] for e in ev});score=max(e['score'] for e in ev)+.14*(dates-1)+(.15 if t4 else 0)
   # A top-level mother is preferred only among actual attachment candidates.
   # Rarity of secondary branches alone never creates an attachment edge.
   score-=.05*min(depth(p),3);candidates.append((score,p,ev,t4,dates))
  candidates.sort(reverse=True,key=lambda x:x[0]);best=candidates[0];score,p,ev,t4,dates=best;h=hier[child];current=h.get('parent_root_id');supported_current=next((x for x in candidates if x[1]==current),None)
  record=dict(root_id=child,previous_parent=current,parent_root_id=p,evidence=ev,score=round(score,4),alternatives=[dict(parent_root_id=x[1],score=round(x[0],4)) for x in candidates[1:]],applied=False,review_status='pending')
  if child in protected_h:record['reason']='uploaded13 user-edited parentage retained';parent_reviews.append(record);continue
  if current==p:record['reason']='current parent has attachment support';parent_reviews.append(record);continue
  if depth(p)>0:record['reason']='new secondary/later-order branch requires origin review; not inferred from rarity or contact';parent_reviews.append(record);continue
  if not t4 or dates<2 or len(candidates)>1 and score-candidates[1][0]<.18 or supported_current and score-supported_current[0]<.25:
   record['reason']='insufficient separation between actual attachment candidates';parent_reviews.append(record);continue
  cur=p;seen={child}
  while cur and cur not in seen:seen.add(cur);cur=hier[cur].get('parent_root_id')
  if cur:record['reason']='would create parent cycle';parent_reviews.append(record);continue
  h['parent_root_id']=p;h['parentage_confirmed']=False;h['review_status']='pending';h['parent_evidence_v58']=ev;record['applied']=True;record['reason']='T4 and earlier dates support one-sided physical attachment';parent_changes.append(record);parent_reviews.append(record)
 # Audit inherited second-order and deeper laterals individually. Do not flatten
 # them just to satisfy a prior: no evidence means pending, not invented parent.
 nested=[]
 for r,h in hier.items():
  if r not in present or depth(r)<2:continue
  p=h.get('parent_root_id');ev=evidence.get((r,p),[]);status='user_edit_retained' if r in protected_h else 'attachment_supported' if any(e['period']=='T4' for e in ev) and len({e['period'] for e in ev})>=2 else 'requires_origin_review'
  nested.append(dict(root_id=r,parent_root_id=p,family_root_id=h['family_root_id'],display_label=h['label'],status=status,evidence=ev,reason='verify whether origin is on this lateral or on the primary mother; secondary branching is not assumed'))
 # Remove only identities with no remaining observations; keep every surviving
 # identity registered through T4 even when the image mask has an observation gap.
 for h in hier.values():
  p=h.get('parent_root_id');seen=set()
  while p and p not in present and p not in seen:seen.add(p);p=hier[p].get('parent_root_id')
  h['parent_root_id']=p if p in present else None
 hier={r:h for r,h in hier.items() if r in present};children=collections.defaultdict(list)
 for r,h in hier.items():children[h.get('parent_root_id')].append(r)
 # Independently new is a separate biological assertion. Exclude continuation
 # and plausible parentage FIRST; boundary entry provides observable evidence.
 independent_audit=[];used={h['label'] for h in hier.values() if h.get('label')};number=max([int(x) for x in used if x.isdigit()]+[0])+1;historical_N={v for h in hier.values() for v in [h.get('label')]+h.get('previous_display_labels_v57',[]) if v and v.startswith('N-') and v[2:].isdigit()};new_independent_number=max([int(v[2:]) for v in historical_N]+[0])+1;geometry_cache=[{r:geometry(g) for r,g in groups.items()} for groups in grouped]
 for r in children[None]:
  h=hier[r];first=present[r][0];g=grouped[first][r];points=np.concatenate([s['points'] for s in g]);endpoint=np.asarray([p for s in g for p in [s['points'][0],s['points'][-1]]]);f=frames[first];boundary=float(np.min(np.c_[endpoint[:,0],endpoint[:,1],f['width']-1-endpoint[:,0],f['height']-1-endpoint[:,1]]))<6;possible_parent=any(c==r for c,p in evidence);previous=[]
  for fi in range(first):
   best=0
   for oldr,geom in geometry_cache[fi].items():
    if (geom['p'].max(axis=0)+6<points.min(axis=0)).any() or (geom['p'].min(axis=0)-6>points.max(axis=0)).any():continue
    best=max(best,similarity({'points':points},geom)[0])
   previous.append(dict(period='T'+str(fi+1),maximum_prior_axis_coverage=round(best,4)))
  native=native_temporal[first];new_fraction=float(np.mean(native[points[:,1]*f['width']+points[:,0]]==1));no_previous=not any(e['maximum_prior_axis_coverage']>.25 for e in previous);confirmed=h.get('independent_new_confirmed_v58',False);safe_new=first>0 and not possible_parent and no_previous and boundary and new_fraction>=.60
  role='baseline_independent' if first==0 else 'independent_new_supported' if safe_new or confirmed else 'unresolved_side_vs_independent'
  h['birth_role_v58']=role;h['independent_new_confirmed_v58']=bool(confirmed);h['independent_new_evidence_v58']=dict(boundary_entry_visible=boundary,new_class_fraction=round(new_fraction,4),plausible_parent_candidates=possible_parent,prior_axis_matches=previous,prior_continuation_excluded=no_previous,fully_human_confirmed=bool(confirmed))
  if safe_new and not h['label'].startswith('N-'):
   oldlabel=h['label'];h['label']='N-'+str(new_independent_number).zfill(4);new_independent_number+=1;used.add(h['label']);h['previous_display_labels_v57']=list(dict.fromkeys(h.get('previous_display_labels_v57',[])+[oldlabel]))
  if h['label'].startswith('N-') and not safe_new and not confirmed and r not in protected_h:
   oldlabel=h['label']
   while str(number).zfill(4) in used:number+=1
   h['label']=str(number).zfill(4);number+=1;used.add(h['label']);h['previous_display_labels_v57']=list(dict.fromkeys(h.get('previous_display_labels_v57',[])+[oldlabel]));h['label_revision_reason_v58']='independent-new assertion withheld until continuation and side-root origin are excluded'
  independent_audit.append(dict(root_id=r,display_label=h['label'],role=role,**h['independent_new_evidence_v58']))
 used=set(hier[r]['label'] for r in children[None]);rename=[]
 def visit(r,family):
  h=hier[r];h['family_root_id']=family;occupied=set()
  for c in sorted(children[r],key=lambda c:(c not in protected_h,hier[c].get('label') or '')):
   ch=hier[c];oldlabel=ch.get('label') or '';kind=ch.get('branch_kind','existing');kind='new' if kind=='new' else 'existing';suffix=oldlabel.split('-')[-1];n=int(suffix.lstrip('N')) if suffix.lstrip('N').isdigit() else 1;prefix='N' if kind=='new' else ''
   if not oldlabel:
    reserved=[]
    for other in children[r]:
     for v in [hier[other].get('label')]+hier[other].get('previous_display_labels_v57',[]):
      if v and v.startswith(h['label']+'-'):
       part=v[len(h['label'])+1:]
       if '-' not in part and part.lstrip('N').isdigit() and part.startswith('N')==(kind=='new'):reserved.append(int(part.lstrip('N')))
    n=max(reserved+[0])+1
   while prefix+str(n) in occupied or h['label']+'-'+prefix+str(n) in used:n+=1
   label=h['label']+'-'+prefix+str(n);occupied.add(prefix+str(n));used.add(label);ch.update(label=label,branch_kind=kind,birth_role_v58='new_lateral' if kind=='new' else 'existing_lateral',independent_new_confirmed_v58=False)
   if label!=oldlabel:ch['previous_display_labels_v57']=list(dict.fromkeys(ch.get('previous_display_labels_v57',[])+[oldlabel]));rename.append(dict(root_id=c,old_label=oldlabel,label=label))
   visit(c,family)
 for r in children[None]:visit(r,r)
 for f in frames:
  for s in f['segments']:
   h=hier[s['root_id']];s.update(display_label=h['label'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'])
 # Stable identity lifetime is distinct from actually observed foreground.
 lifetime=[];gaps=[]
 for r,h in hier.items():
  obs=present[r];first=min(obs);h['first_seen_period']='T'+str(first+1);h['observed_periods']=['T'+str(fi+1) for fi in obs];states=[]
  for fi in range(4):
   if fi<first:status='before_first_observation'
   elif fi in obs:status='observed_same_identity'
   elif any(j>fi for j in obs):status='observation_gap_identity_retained'
   else:status='not_reliably_matched_identity_retained'
   states.append(dict(period='T'+str(fi+1),status=status,root_id=r,label=h['label'],parent_root_id=h.get('parent_root_id'),observed=fi in obs))
   if fi>=first and fi not in obs:gaps.append(dict(root_id=r,display_label=h['label'],period='T'+str(fi+1),status=status,reference_period='T'+str(max(j for j in obs if j<fi)+1),point=grouped[max(j for j in obs if j<fi)][r][0]['points'][0]))
  lifetime.append(dict(root_id=r,label=h['label'],parent_root_id=h.get('parent_root_id'),birth_role=h.get('birth_role_v58'),states=states,identity_retained_through_T4=True,visibility_is_not_identity=True))
 st['root_track_hierarchy']=list(hier.values());st['root_lifetime_registry_v58']=lifetime;st['root_tracks']=[dict(root_id=r,display_label=h['label'],first_seen_period=h['first_seen_period'],observed_periods=h['observed_periods'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'],review_status='pending',biological_identity_confirmed=False) for r,h in hier.items()];st['cross_period_links']=[]
 for r,h in hier.items():
  obs=[(f,[s for s in f['segments'] if s['root_id']==r]) for f in frames];obs=[(f,g) for f,g in obs if g]
  for (a,ga),(b,gb) in zip(obs,obs[1:]):st['cross_period_links'].append(dict(id='L-'+r+'-'+b['period'],previous_period=a['period'],current_period=b['period'],root_id=r,display_label=h['label'],previous_segment=ga[0]['id'],current_segment=gb[0]['id'],previous_segments=[s['id'] for s in ga],current_segments=[s['id'] for s in gb],kind='same_root_candidate',review_status='pending'))
 audit=dict(source_upload='root_annotation_progress (13).json',source_SHA256=hashlib.sha256(raw).hexdigest(),priority='side-root origin first, identity continuity second; no second birth after observation gap',user_edited_segments_retained=len(protected),user_edited_hierarchy_retained=sorted(protected_h),continuation_recoveries=merges,tip_extension_identity_joins=growth,distinct_lateral_identity_splits=distinct_laterals,uncertain_continuations=uncertain,parent_changes=parent_changes,parent_candidates=parent_reviews,nested_lateral_reviews=nested,independent_new_reviews=independent_audit,visibility_gaps=gaps,uncertain_junctions=junction_review,renames=rename,missing_foreground_fabricated=False)
 st['parentage_revision_v58']=audit;st['human_reviewed']=False;st['status']='proposal_under_review'
 assert d['images']==original['images'];assert len({h['label'] for h in hier.values()})==len(hier)
 for f,o in zip(frames,original['root_structure_annotation']['periods']):
  by={s['id']:s for s in f['segments']};assert f['owner_map_rle']==o['owner_map_rle'];assert [s['points'] for s in f['segments']]==[s['points'] for s in o['segments']]
  for s in o['segments']:
   if (f['period'],s['id']) in protected:assert by[s['id']]['root_id']==s['root_id']
 for r in protected_h:
  if r in hier:assert hier[r].get('parent_root_id')==next(h for h in original['root_structure_annotation']['root_track_hierarchy'] if h['root_id']==r).get('parent_root_id')
 write('root_structure_draft.json',d);write('revision_audit.json',audit);stats=dict(passed=True,source_SHA256=hashlib.sha256(raw).hexdigest(),all_native_pixel_layers_unchanged=True,all_native_geometry_and_owner_maps_unchanged=True,manual_edit_segments_preserved=len(protected),manual_hierarchy_roots_preserved=len(protected_h),continuation_recoveries=len(merges),tip_extension_joins=len(growth),distinct_lateral_identity_splits=len(distinct_laterals),parent_changes=len(parent_changes),nested_lateral_review_count=len(nested),nested_unresolved=sum(x['status']=='requires_origin_review' for x in nested),independent_new_supported=sum(x['role']=='independent_new_supported' for x in independent_audit),independent_new_withheld=sum(x['role']=='unresolved_side_vs_independent' for x in independent_audit),identity_lifetime_registry_size=len(lifetime),all_lifetimes_have_T4_identity=True,visibility_gap_records=len(gaps),missing_root_pixels_fabricated=False)
 write('draft_verification.json',stats);print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
