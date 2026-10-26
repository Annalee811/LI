"""Auditable multi-date identity proposals; never changes the four pixel layers."""
from pathlib import Path
import collections,copy,hashlib,json,math,sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_continuity_review_v56_20261026'
SOURCE=OUT/'source_progress_12.json'
PREV=ROOT/'results/B3_root_id_edit_fix_v55_20261025/root_structure_draft.json'
def write(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
def tangent(points):
 p=np.asarray(points,float);i=np.arange(len(p));v=p[np.minimum(i+7,len(p)-1)]-p[np.maximum(i-7,0)];v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-9);return v
def point_data(frame,protected):
 pp=[];tt=[];rr=[];ww=[]
 for s in frame['segments']:
  p=np.asarray(s['points'],float);pp.extend(p);tt.extend(tangent(p));rr.extend([s['root_id']]*len(p));ww.extend([5. if (frame['period'],s['id']) in protected else 1.]*len(p))
 p=np.asarray(pp);return dict(p=p,t=np.asarray(tt),r=np.asarray(rr),w=np.asarray(ww),tree=cKDTree(p))
def vote(segment,sources):
 p=np.asarray(segment['points'],float);t=tangent(p);votes=[collections.Counter() for _ in p]
 for data,period_weight in sources:
  dist,idx=data['tree'].query(p,k=12,distance_upper_bound=9)
  valid=idx<len(data['p']);idx=np.minimum(idx,len(data['p'])-1)
  alignment=np.abs(np.sum(t[:,None,:]*data['t'][idx],axis=2));score=np.exp(-dist**2/32)*alignment**6*data['w'][idx]*period_weight
  score[~valid|(alignment<.83)]=0
  for i in range(len(p)):
   best={}
   for j in range(idx.shape[1]):
    if score[i,j]>.03:
     rid=str(data['r'][idx[i,j]]);best[rid]=max(best.get(rid,0),float(score[i,j]))
   votes[i].update(best)
 return votes
def main():
 sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
 d=json.loads(SOURCE.read_text());st=d['root_structure_annotation'];assert require_active_position(st['position'])=='0353-2-B3'
 original=copy.deepcopy(d);old=json.loads(PREV.read_text())['root_structure_annotation'];frames=st['periods'];hier={h['root_id']:h for h in st['root_track_hierarchy']}
 protected=set()
 for f,o in zip(frames,old['periods']):
  by={s['id']:s for s in o['segments']}
  for s in f['segments']:
   before=by.get(s['id']);
   label=hier[s['root_id']]['label']
   if f['period']=='T1' or f['period']!='T4' and (label=='0001' or label.startswith('0001-') or not before or before['root_id']!=s['root_id'] or before['points']!=s['points'] or s.get('review_status')=='confirmed'): protected.add((f['period'],s['id']))
 # Anchor labels typed by the user are authoritative, independent of stale registry dates.
 protected_hierarchy={h['root_id'] for h in hier.values() if h.get('review_status') in ['modified','confirmed'] or h['label']=='0001' or h['label'].startswith('0001-')}
 original_parents={rid:h.get('parent_root_id') for rid,h in hier.items()}
 all_changes=[];uncertain=[]
 # Two passes: transfer reviewed T1--T3 identities to T4, then T4 evidence to
 # untouched fragments in earlier dates. Always compare tangents, not color.
 for passnum,targets in enumerate([[3],[0,1,2,3]]):
  data=[point_data(f,protected) for f in frames]
  for fi in targets:
   f=frames[fi];plans=[]
   for s in f['segments']:
    if (f['period'],s['id']) in protected:continue
    sources=[(data[j],(1.35 if j==3 else 1.)/(1+.18*abs(fi-j))) for j in range(4) if j!=fi]
    vv=vote(s,sources);total=sum(vv,collections.Counter());rank=total.most_common(3)
    if not rank:continue
    top,score=rank[0];mass=sum(total.values());coverage=sum(bool(v) for v in vv)/len(vv);share=score/max(1e-9,mass);runner=rank[1][1] if len(rank)>1 else 0
    # Point-level partitions are allowed only where *reviewed* identities give
    # sustained evidence. A perpendicular crossing cannot supply votes.
    labels=[]
    for v in vv:
     r=v.most_common(2)
     if r and r[0][0]!=s['root_id'] and r[0][1]>=1.2 and (len(r)==1 or r[0][1]>1.65*r[1][1]):labels.append(r[0][0])
     else:labels.append(s['root_id'])
    # If an entire short trace matches unambiguously, keep one identity.
    if top!=s['root_id'] and coverage>=.65 and share>=.72 and score>2*runner and len(s['points'])>=8:
     labels=[top]*len(vv)
    else:
     encoded={r:i for i,r in enumerate(sorted(set(labels)))};reverse={i:r for r,i in encoded.items()}
     arr=np.array([encoded[r] for r in labels]); # categorical *mode*, never a numeric median
     sm=[]
     for i in range(len(arr)):
      lo=max(0,i-7);hi=min(len(arr),i+8);c=collections.Counter(arr[lo:hi]);sm.append(reverse[c.most_common(1)[0][0]])
     labels=sm
    runs=[]
    for i,r in enumerate(labels):
     if not runs or runs[-1]['rid']!=r:runs.append(dict(rid=r,start=i,end=i+1))
     else:runs[-1]['end']=i+1
    # Merge tiny islands back into their original identity; do not invent roots.
    for run in runs:
     if run['end']-run['start']<16 and run['rid']!=s['root_id']:
      labels[run['start']:run['end']]=[s['root_id']]*(run['end']-run['start'])
    if any(r!=s['root_id'] for r in labels):plans.append((s,labels,dict(coverage=round(coverage,3),share=round(share,3),candidates=rank)))
    elif top!=s['root_id'] and coverage>.25 and share>.4:uncertain.append(dict(period=f['period'],segment=s['id'],root_id=s['root_id'],candidate=top,coverage=round(coverage,3),share=round(share,3)))
   apply_plans(f,plans,passnum,all_changes)
 canonical_manual_overrides=[];overridden_keys=set()
 # Resolve identity flicker against the earliest visible axis. This is a
 # per-point geometric comparison, NOT a forced temporal-class grammar.
 # A manually edited later segment can still contain a wrong mother/side ID;
 # override it only on a sustained, close and tangent-consistent early axis.
 for fi in [1,2,3]:
  f=frames[fi];sources=[point_data(frames[j],set()) for j in range(fi)];plans=[]
  for s in f['segments']:
   if s.get('review_status')=='confirmed':continue
   p=np.asarray(s['points']);t=tangent(p);labels=[s['root_id']]*len(p);claimed=np.zeros(len(p),bool)
   for data in sources:
    dist,ix=data['tree'].query(p,k=20,distance_upper_bound=7);valid=ix<len(data['p']);ix=np.minimum(ix,len(data['p'])-1);angle=np.abs(np.sum(t[:,None,:]*data['t'][ix],axis=2));score=np.exp(-dist**2/18)*angle**10;score[~valid|(angle<.93)]=0
    for i in range(len(p)):
     if claimed[i]:continue
     candidates={}
     for j in range(ix.shape[1]):
      if score[i,j]>.2:
       r=str(data['r'][ix[i,j]]);candidates[r]=max(candidates.get(r,0),float(score[i,j]))
     rank=sorted(candidates.items(),key=lambda x:-x[1])
     if rank and (len(rank)==1 or rank[0][1]>1.8*rank[1][1]):labels[i]=rank[0][0];claimed[i]=True
   original_labels=labels[:]
   for i in range(len(labels)):
    c=collections.Counter(original_labels[max(0,i-7):min(len(labels),i+8)]);labels[i]=c.most_common(1)[0][0]
   runs=[]
   for i,r in enumerate(labels):
    if not runs or runs[-1][0]!=r:runs.append([r,i,i+1])
    else:runs[-1][2]=i+1
   for r,a,b in runs:
    if b-a<18 and r!=s['root_id']:labels[a:b]=[s['root_id']]*(b-a)
   matched=collections.Counter(r for r,c in zip(labels,claimed) if c);rank=matched.most_common()
   if rank and rank[0][1]/len(labels)>.75:labels=[rank[0][0]]*len(labels)
   if any(r!=s['root_id'] for r in labels):
    info=dict(coverage=round(float(claimed.mean()),3),share=round((rank[0][1]/max(1,claimed.sum())) if rank else 0,3),candidates=rank[:3],criterion='earliest visible axis within 7px, tangent >=0.93, unique candidate >=1.8 margin, sustained run')
    plans.append((s,labels,info))
    if (f['period'],s['id']) in protected:
     overridden_keys.add((f['period'],s['id']));canonical_manual_overrides.append(dict(period=f['period'],segment_id=s['id'],previous_root_id=s['root_id'],proposed_root_ids=sorted(set(labels)),**info))
  apply_plans(f,plans,3,all_changes)
 lateral_splits=[]
 # Separate a newly visible one-sided lateral that still shares the mother's
 # ID. Require a non-main arm, new-class support, absence from T1, and an
 # aligned T4 counterpart. Reviewed edits and crossings remain untouched.
 t1tree=cKDTree(np.concatenate([np.asarray(s['points']) for s in frames[0]['segments']]))
 for fi in [1,2]:
  f=frames[fi];by={s['id']:s for s in f['segments']};temporal=decode(d['images'][fi]['layers']['temporal'])
  for n in f['nodes']:
   aa=n['arms']
   if len(aa)!=3 or any('direction' not in a for a in aa):continue
   dot,i,j=min((float(np.dot(a['direction'],b['direction'])),i,j) for i,a in enumerate(aa) for j,b in enumerate(aa) if i<j);k=3-i-j
   main=[by[aa[i]['segment_id']],by[aa[j]['segment_id']]];side=by[aa[k]['segment_id']];rid=side['root_id']
   if dot>-.9 or any(s['root_id']!=rid for s in main) or side['id'] in [s['id'] for s in main] or (f['period'],side['id']) in protected:continue
   pts=np.asarray(side['points']);ed=min(np.linalg.norm(pts[0]-[n['x'],n['y']]),np.linalg.norm(pts[-1]-[n['x'],n['y']]))
   pixel=temporal[pts[:,1]*f['width']+pts[:,0]];nf=float(np.mean(pixel==1));overlap=float(np.mean(t1tree.query(pts)[0]<7))
   if ed>10 or side['length_px']<40 or nf<.8 or overlap>.3:continue
   source=dict(p=pts,t=tangent(pts),r=np.asarray(['CHILD']*len(pts)),w=np.ones(len(pts))*5,tree=cKDTree(pts));plans={fi:[(side,['CHILD']*len(pts),dict(coverage=1.,share=1.,candidates=[['CHILD',1.]]))]}
   for fj in range(fi+1,4):
    matches=[]
    for candidate in frames[fj]['segments']:
     if candidate['root_id']!=rid or (frames[fj]['period'],candidate['id']) in protected:continue
     votes=vote(candidate,[(source,1.)]);covered=np.asarray([bool(v) for v in votes]);coverage=float(np.mean(covered))
     if coverage>=.55 and covered.sum()>=15:matches.append((candidate,['CHILD']*len(covered),dict(coverage=round(coverage,3),share=1.,candidates=[['CHILD',1.]])))
    if matches:plans[fj]=matches
   if 3 not in plans:continue
   newrid='R'+str(max(int(h[1:]) for h in hier)+1).zfill(4)
   for fj,entries in plans.items():
    entries=[(s,[newrid]*len(labels),info) for s,labels,info in entries];apply_plans(frames[fj],entries,2,all_changes)
   hier[newrid]=dict(root_id=newrid,label='TEMP-N1',parent_root_id=rid,family_root_id=hier[rid]['family_root_id'],branch_kind='new',review_status='pending',parentage_confirmed=False,creation_reason='new_lateral_separated_from_shared_mother_identity',parent_evidence_v56=[dict(period=f['period'],node=n['id'],point=[n['x'],n['y']],new_class_fraction=nf,T1_axis_overlap=overlap,T4_geometry_supported=True)])
   original_parents[newrid]=None;lateral_splits.append(dict(root_id=newrid,mother_root_id=rid,first_period=f['period'],seed_segment=side['id'],observed_periods=[frames[j]['period'] for j in plans],new_class_fraction=nf,T1_axis_overlap=overlap,review_status='pending'))
 for fi,f in enumerate(frames):
  temporal=decode(d['images'][fi]['layers']['temporal'])
  for s in f['segments']:
   points=np.rint(s['points']).astype(int);values=temporal[points[:,1]*f['width']+points[:,0]]
   s['semantic_code_counts']={str(int(k)):int(v) for k,v in zip(*np.unique(values,return_counts=True))}
 # Junction identities must follow reliable root continuation, never merge a
 # lateral with the main axis solely because their foreground is connected.
 # Existing reviewed T1--T3 assignments are kept exactly; unresolved topology
 # is listed rather than recolored or silently forcing a family assignment.
 parent_changes=[];evidence=collections.defaultdict(list)
 for fi in [3,2,1,0]:
  f=frames[fi];by={s['id']:s for s in f['segments']}
  for n in f['nodes']:
   if n.get('review_status') in ['modified','confirmed'] or len(n['arms'])!=3:continue
   arms=[a for a in n['arms'] if a.get('direction') and a['segment_id'] in by]
   if len(arms)!=3:continue
   pairs=sorted([(float(np.dot(a['direction'],b['direction'])),i,j) for i,a in enumerate(arms) for j,b in enumerate(arms) if i<j])
   dot,i,j=pairs[0];k=3-i-j;a,b,c=arms[i],arms[j],arms[k];pa,pb,ch=[by[x['segment_id']]['root_id'] for x in [a,b,c]]
   if dot>-.86 or pairs[1][0]-dot<.15 or pa!=pb or ch==pa:continue
   sp=by[c['segment_id']];center=np.array([n['x'],n['y']]);direction=np.asarray(c['direction']);p=np.asarray(sp['points']);ed=min(np.linalg.norm(p[0]-center),np.linalg.norm(p[-1]-center));side=abs(np.dot(direction,a['direction']))
   near=np.concatenate([np.asarray(x['points']) for x in f['segments'] if x['root_id']==ch]);near=near[np.linalg.norm(near-center,axis=1)<50];projection=(near-center)@direction
   if ed>10 or side>.90 or np.count_nonzero(projection<-8)>max(3,.12*np.count_nonzero(projection>8)):continue
   evidence[ch,pa].append(dict(period=f['period'],node=n['id'],point=[n['x'],n['y']],opposition=round(-dot,4),endpoint_distance=round(float(ed),3),review_status='pending',basis='three-arm one-sided lateral, tangent-consistent mother; T4 preferred, RGB contact requires review'))
 # Metadata is regenerated from actual segment observations, not inherited dates.
 present=collections.defaultdict(list)
 for f in frames:
  for rid in {s['root_id'] for s in f['segments']}:present[rid].append(f['period'])
 for (child,pa),ev in sorted(evidence.items(),key=lambda x: (not any(e['period']=='T4' for e in x[1]),-len(x[1]))):
  if child in protected_hierarchy or child not in present or pa not in present:continue
  if not any(e['period']=='T4' for e in ev) or len({e['period'] for e in ev})<2:continue
  h=hier[child];alternatives=[(p,v) for (c,p),v in evidence.items() if c==child and p!=pa]
  # A long trace may pass behind another root even when the binary graph has
  # only three visible arms. Do not attach such an independent axis to a family.
  t4child=[s for s in frames[3]['segments'] if s['root_id']==child]
  attach=np.asarray(next(e['point'] for e in ev if e['period']=='T4'))
  allp=np.concatenate([np.asarray(s['points']) for s in t4child])
  if np.linalg.norm(allp-attach,axis=1).max()>260 and not h.get('parent_root_id'):continue
  if alternatives and max(len(v) for p,v in alternatives)>=len(ev):continue
  cur=pa;seen={child}
  while cur and cur not in seen:seen.add(cur);cur=hier[cur].get('parent_root_id')
  if cur or min(present[pa])>min(present[child]):continue
  if h.get('parent_root_id')==pa:continue
  before=h.get('parent_root_id');h['parent_root_id']=pa;h['parent_evidence_v56']=ev;h['review_status']='pending';h['parentage_confirmed']=False
  h['branch_kind']='new' if min(present[child])>'T1' else 'existing';parent_changes.append(dict(root_id=child,previous_parent=before,parent=pa,evidence=ev))
 # Remove only genuinely unused records; keep parents with observed descendants
 # by attaching descendants to their nearest surviving ancestor.
 for h in hier.values():
  while h.get('parent_root_id') and h['parent_root_id'] not in present:h['parent_root_id']=hier[h['parent_root_id']].get('parent_root_id')
 hier={rid:h for rid,h in hier.items() if rid in present}
 renames=[];used=set();heads=sorted([h for h in hier.values() if not h.get('parent_root_id')],key=lambda h:(min(present[h['root_id']]),h['label']))
 new_counter=0
 for h in heads:
  prior=h['label'];is_later=min(present[h['root_id']])>'T1'
  # N denotes first visible after T1, not proof of biological birth. User-typed
  # family labels are retained; N is for independent roots, never laterals.
  rid=h['root_id'];firstfi=int(min(present[rid])[1:])-1
  first_segments=[s for s in frames[firstfi]['segments'] if s['root_id']==rid];codes=collections.Counter()
  for s in first_segments:codes.update({int(k):v for k,v in s.get('semantic_code_counts',{}).items()})
  pts=np.concatenate([np.asarray(s['points']) for s in first_segments]);earlier=np.concatenate([np.asarray(s['points']) for f in frames[:firstfi] for s in f['segments']]) if firstfi else None
  previous_overlap=float(np.mean(cKDTree(earlier).query(pts)[0]<7)) if firstfi else 1.
  new_fraction=codes[1]/max(1,sum(codes.values()));new_candidate=is_later and new_fraction>=.6 and sum(s['length_px'] for s in first_segments)>=50 and previous_overlap<.25
  h['independent_new_evidence_v56']=dict(first_seen_period=min(present[rid]),new_class_fraction=round(new_fraction,4),earlier_axis_overlap=round(previous_overlap,4),status='new_independent_candidate' if new_candidate else 'unresolved_identity_not_assumed_new' if is_later else 'T1_existing')
  if new_candidate:
   new_counter+=1;h['label']='N-'+str(new_counter).zfill(4)
  elif '-' in prior and not prior.startswith('N-'):
   i=1
   while str(i).zfill(4) in used or any(x['label']==str(i).zfill(4) for x in heads):i+=1
   h['label']=str(i).zfill(4)
  used.add(h['label']);h['branch_kind']='independent';h['family_root_id']=h['root_id']
  if h['label']!=prior:renames.append(dict(root_id=h['root_id'],old_label=prior,label=h['label'],reason='independent root first observed after T1' if is_later else 'independent root'))
 children=collections.defaultdict(list)
 for h in hier.values():
  if h.get('parent_root_id'):children[h['parent_root_id']].append(h)
 def visit(h):
  assigned=set()
  # Preserve user-entered suffixes first. Parent change cascades the prefix only.
  for c in sorted(children[h['root_id']],key=lambda c:(c['root_id'] not in protected_hierarchy,c['label'])):
   oldlabel=c['label'];suffix=oldlabel.split('-')[-1] if original_parents.get(c['root_id'])==h['root_id'] else ('N1' if c['branch_kind']=='new' else '1')
   if not suffix.lstrip('N').isdigit():suffix='1'
   if c['branch_kind']=='new' and not suffix.startswith('N'):suffix='N'+suffix
   if c['branch_kind']!='new':suffix=suffix.lstrip('N')
   num=int(suffix.lstrip('N'));prefix='N' if c['branch_kind']=='new' else ''
   while prefix+str(num) in assigned or h['label']+'-'+prefix+str(num) in used:num+=1
   suffix=prefix+str(num);assigned.add(suffix);c['label']=h['label']+'-'+suffix;c['family_root_id']=h['family_root_id'];used.add(c['label'])
   if c['label']!=oldlabel:renames.append(dict(root_id=c['root_id'],old_label=oldlabel,label=c['label'],reason='mother hierarchy prefix or unique side suffix'))
   visit(c)
 for h in heads:visit(h)
 for rid,h in hier.items():
  h['observed_periods']=sorted(present[rid]);h['first_seen_period']=min(present[rid]);h['first_observation_is_not_proven_birth']=True
 for f in frames:
  for s in f['segments']:
   h=hier[s['root_id']];s.update(display_label=h['label'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'])
 st['root_track_hierarchy']=list(hier.values());st['root_tracks']=[dict(root_id=rid,display_label=h['label'],first_seen_period=h['first_seen_period'],observed_periods=h['observed_periods'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'],review_status=h['review_status'],biological_identity_confirmed=False) for rid,h in hier.items()]
 st['cross_period_links']=[]
 for rid,h in hier.items():
  observed=[(f,[s for s in f['segments'] if s['root_id']==rid]) for f in frames];observed=[(f,g) for f,g in observed if g]
  for (a,ga),(b,gb) in zip(observed,observed[1:]):st['cross_period_links'].append(dict(id='L-'+rid+'-'+b['period'],previous_period=a['period'],current_period=b['period'],root_id=rid,display_label=h['label'],previous_segment=ga[0]['id'],current_segment=gb[0]['id'],previous_segments=[s['id'] for s in ga],current_segments=[s['id'] for s in gb],kind='same_root_candidate',review_status='pending'))
 conflicts=[]
 for f in frames:
  by={s['id']:s for s in f['segments']}
  for n in f['nodes']:
   arms=n['arms'];rr=[by[a['segment_id']]['root_id'] for a in arms]
   if len(arms)==3 and len(set(rr))==1:
    conflicts.append(dict(period=f['period'],node_id=n['id'],point=[n['x'],n['y']],root_id=rr[0],display_label=hier[rr[0]]['label'],reason='three visible arms share one identity: review lateral versus crossing/curve, do not assume birth',review_status='pending'))
 # Deduplicate evidence repeated across matching passes; unresolved proposals
 # are not materialized as artificial roots or labelled confirmed.
 uncertain=list({(u['period'],u['segment'],u['candidate']):u for u in uncertain}.values())
 st['continuity_revision_v56']=dict(source_SHA256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),protected_manual_segment_count=len(protected),canonical_manual_overrides=canonical_manual_overrides,changes=all_changes,parent_changes=parent_changes,lateral_splits=lateral_splits,renames=renames,uncertain_matches=uncertain,same_id_junction_reviews=conflicts,T4_role='Retrospective geometry and one-sided attachment evidence, not proof of earlier visibility',new_independent_prefix='N-0001',no_pixel_label_changes=True)
 st['status']='proposal_under_review';st['human_reviewed']=False
 assert d['images']==original['images']
 for fi,f in enumerate(frames):
  by={s['id']:s for s in f['segments']}
  for s in original['root_structure_annotation']['periods'][fi]['segments']:
   if (f['period'],s['id']) in protected and (f['period'],s['id']) not in overridden_keys:assert by[s['id']]['root_id']==s['root_id'] and by[s['id']]['points']==s['points']
  assert decode(f['owner_map_rle']).shape==(f['width']*f['height'],)
  assert np.array_equal(decode(f['owner_map_rle'])>0,decode(original['root_structure_annotation']['periods'][fi]['owner_map_rle'])>0)
 write('root_structure_draft.json',d);write('revision_audit.json',st['continuity_revision_v56'])
 stats=dict(passed=True,source_SHA256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),pixel_layers_unchanged=True,T1_root_assignments_unchanged=True,protected_manual_segment_count=len(protected),canonical_manual_conflict_proposals=len(canonical_manual_overrides),other_protected_assignments_unchanged=True,identity_changes=len(all_changes),parent_changes=len(parent_changes),new_lateral_splits=len(lateral_splits),renames=len(renames),unresolved_matches=len(uncertain),same_id_junction_reviews=len(conflicts),family0001_observations={h['label']:h['observed_periods'] for h in hier.values() if h['label']=='0001' or h['label'].startswith('0001-')},independent_N_roots=new_counter)
 write('draft_verification.json',stats);print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
def decode(rle):return np.repeat(np.asarray(rle[::2],np.int32),rle[1::2])
def encode(a):
 idx=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,len(a)];return np.column_stack((a[idx[:-1]],np.diff(idx))).ravel().tolist()
def apply_plans(f,plans,passnum,changes):
 if not plans:return
 owner=decode(f['owner_map_rle']);nextindex=max(s['index'] for s in f['segments'])+1;segments=[];replacements={}
 planned={s['id']:(labels,info) for s,labels,info in plans}
 for s in f['segments']:
  if s['id'] not in planned:segments.append(s);continue
  labels,info=planned[s['id']];runs=[]
  for p,r in zip(s['points'],labels):
   if not runs or runs[-1][0]!=r:runs.append([r,[]])
   runs[-1][1].append(p)
  pieces=[]
  for i,(r,pts) in enumerate(runs):
   piece=copy.deepcopy(s)
   if i:piece['index']=nextindex;piece['id']=f['period']+'-S'+str(nextindex).zfill(4);nextindex+=1
   piece['root_id']=r;piece['points']=pts;piece['length_px']=round(float(np.linalg.norm(np.diff(np.asarray(pts),axis=0),axis=1).sum()),3);piece['review_status']='pending';piece['continuity_proposal_v56']=dict(previous_root_id=s['root_id'],pass_index=passnum,**info)
   pieces.append(piece);segments.append(piece)
  replacements[s['id']]=pieces
  if len(pieces)>1:
   loc=np.flatnonzero(owner==s['index']);coords=np.column_stack((loc%f['width'],loc//f['width']));dist=np.stack([cKDTree(p['points']).query(coords)[0] for p in pieces]);winner=dist.argmin(axis=0)
   owner[loc]=np.asarray([p['index'] for p in pieces])[winner]
  changes.append(dict(period=f['period'],segment_id=s['id'],from_root_id=s['root_id'],to_parts=[dict(segment_id=p['id'],root_id=p['root_id'],point_count=len(p['points'])) for p in pieces],**info))
 def nearest(pieces,x,y):return min(pieces,key=lambda p:float(cKDTree(p['points']).query([x,y])[0]))['id']
 for n in f['nodes']:
  if n.get('owner_segment_id') in replacements:n['owner_segment_id']=nearest(replacements[n['owner_segment_id']],n['x'],n['y'])
  for a in n['arms']:
   if a['segment_id'] in replacements:a['segment_id']=nearest(replacements[a['segment_id']],a['x'],a['y'])
 f['segments']=segments;f['owner_map_rle']=encode(owner)
if __name__=='__main__':main()
