"""Candidate lateral-root families with lifetime IDs; preserve all pixel labels."""
import collections,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_root_family_review_v54_20261025'
PREV=ROOT/'results/B3_editor_interaction_v53_20261025'

def groups(frame):
    result={}
    for s in frame['segments']:result.setdefault(s['root_id'],[]).append(s)
    return result

def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    doc=json.loads((PREV/'root_structure_draft.json').read_text());st=doc['root_structure_annotation'];before=copy.deepcopy(st)
    assert require_active_position(st['position'])=='0353-2-B3'
    periods=st['periods'];first={};lengths=collections.Counter()
    for fi,f in enumerate(periods):
        for s in f['segments']:
            first.setdefault(s['root_id'],fi);lengths[s['root_id']]+=s['length_px']
    # Extra joins require an actual through-connection at a junction. Never join
    # an unpaired lateral arm just because it touches the main root.
    parent={rid:rid for rid in first}
    def find(rid):
        while parent[rid]!=rid:parent[rid]=parent[parent[rid]];rid=parent[rid]
        return rid
    joins=[]
    for f in periods:
        byid={s['id']:s for s in f['segments']}
        for n in f['nodes']:
            if len(n['arms'])<3:continue
            for aa,bb in n['connections']:
                a=next(x for x in n['arms'] if x['id']==aa);b=next(x for x in n['arms'] if x['id']==bb)
                if 'direction' not in a or 'direction' not in b:continue
                dot=float(np.dot(a['direction'],b['direction']))
                if dot>-.91:continue
                alternatives=[float(np.dot(a['direction'],x['direction'])) for x in n['arms'] if x['id'] not in [aa,bb] and 'direction' in x]
                alternatives += [float(np.dot(b['direction'],x['direction'])) for x in n['arms'] if x['id'] not in [aa,bb] and 'direction' in x]
                if alternatives and min(alternatives)-dot<.18:continue
                ca,cb=a.get('original_temporal_code'),b.get('original_temporal_code')
                if ca!=cb and {ca,cb}!={2,3}:continue
                ra,rb=find(byid[a['segment_id']]['root_id']),find(byid[b['segment_id']]['root_id'])
                if ra==rb:continue
                # Do not fuse traces that already form a proposed mother/side pair.
                if any((s['root_id'] in [ra,rb] and s.get('parent_root_id') in [ra,rb]) for s in f['segments']):continue
                keep,drop=sorted([ra,rb],key=lambda rid:(first[rid],-lengths[rid],rid));parent[drop]=keep
                joins.append(dict(period=f['period'],junction=n['id'],kept=keep,merged=drop,reason='unambiguous opposing main-axis arms; not the unpaired side arm'))
    for f in periods:
        for s in f['segments']:
            old=s['root_id'];s['root_id']=find(old)
            if old!=s['root_id']:s['identity_assignment']['continuation_joined_from']=old
            if s.get('parent_root_id'):s['parent_root_id']=find(s['parent_root_id'])
            if s.get('parent_root_id')==s['root_id']:s['parent_root_id']=None
            for c in s.get('previous_candidates',[]):c['previous_root_id']=find(c['previous_root_id'])
    tracks={};frame_groups=[]
    for fi,f in enumerate(periods):
        gg=groups(f);frame_groups.append(gg)
        for rid,segments in gg.items():
            t=tracks.setdefault(rid,dict(root_id=rid,first_index=fi,observed_periods=[],total_length=0))
            t['observed_periods'].append(f['period']);t['total_length']+=sum(s['length_px'] for s in segments)
    evidence=collections.defaultdict(list);unresolved=[]
    for fi,f in enumerate(periods):
        byid={s['id']:s for s in f['segments']}
        for n in f['nodes']:
            # A four-arm crossing is never treated as a lateral emergence.
            if len(n['arms'])!=3 or len(n['connections'])!=1:continue
            aa,bb=n['connections'][0];a=next(x for x in n['arms'] if x['id']==aa);b=next(x for x in n['arms'] if x['id']==bb);side=next(x for x in n['arms'] if x['id'] not in [aa,bb])
            if not all('direction' in x for x in [a,b,side]):continue
            dot=float(np.dot(a['direction'],b['direction']))
            pa,pb=byid[a['segment_id']]['root_id'],byid[b['segment_id']]['root_id'];child=byid[side['segment_id']]['root_id']
            if pa!=pb or child==pa or dot>-.72:continue
            if tracks[pa]['first_index']>tracks[child]['first_index']:
                unresolved.append(dict(period=f['period'],node=n['id'],child=child,parent=pa,reason='parent observed later than child; relation unresolved'));continue
            sg=byid[side['segment_id']];points=np.asarray(sg['points'],float);center=np.array([n['x'],n['y']]);direction=np.asarray(side['direction'])
            endpoint_distance=min(float(np.linalg.norm(points[0]-center)),float(np.linalg.norm(points[-1]-center)))
            local=[]
            for segment in frame_groups[fi][child]:
                p=np.asarray(segment['points'],float);distance=np.linalg.norm(p-center,axis=1);local.extend(p[distance<45].tolist())
            local=np.asarray(local)
            if not len(local):continue
            projection=(local-center)@direction
            backward=int((projection < -8).sum());forward=int((projection>8).sum())
            sideways=abs(float(np.dot(direction,a['direction'])))
            # A root continuing on both sides is a crossing, even when the
            # binary graph happens to have only three visible arms.
            if endpoint_distance>13 or backward>max(3,.15*forward) or sideways>.94:
                unresolved.append(dict(period=f['period'],node=n['id'],child=child,parent=pa,reason='not a convincing one-sided terminating lateral; could be crossing'));continue
            score=.45+.25*max(0,(-dot-.72)/.28)+.15*max(0,1-endpoint_distance/13)+.15*max(0,1-sideways)
            evidence[child,pa].append(dict(period=f['period'],node_id=n['id'],x=n['x'],y=n['y'],score=round(score,4),
                main_arms=[aa,bb],lateral_arm=side['id'],endpoint_distance=round(endpoint_distance,2),
                lateral_direction=side['direction'],
                criterion='Three arms, main root passes through, lateral ends at junction, no opposite-side continuation; RGB review still required'))
    # Recover side-root identities fragmented by the original forward matcher.
    # A shared mother-root attachment plus tangent provides evidence independent
    # of whether that side root is currently called new, persistent or extending.
    attachment_joins=[]
    items=list(evidence.items())
    for i,((ca,pa),ea) in enumerate(items):
        for (cb,pb),eb in items[i+1:]:
            a,b=find(ca),find(cb)
            if a==b or find(pa)!=find(pb) or a==find(pa) or b==find(pa):continue
            if set(tracks[a]['observed_periods'])&set(tracks[b]['observed_periods']):continue
            matches=[]
            for x in ea:
                for y in eb:
                    gap=abs(int(x['period'][1:])-int(y['period'][1:]))
                    distance=float(np.hypot(x['x']-y['x'],x['y']-y['y']))
                    alignment=float(np.dot(x['lateral_direction'],y['lateral_direction']))
                    if gap<=2 and distance<=8 and alignment>=.85:matches.append((distance,alignment,x,y))
            if not matches:continue
            keep,drop=sorted([a,b],key=lambda rid:(tracks[rid]['first_index'],rid));parent[drop]=keep
            tracks[keep]['observed_periods']=sorted(set(tracks[keep]['observed_periods']+tracks[drop]['observed_periods']))
            x=min(matches,key=lambda q:q[0]);attachment_joins.append(dict(kept=keep,merged=drop,mother_root=find(pa),attachment_distance=round(x[0],2),direction_agreement=round(x[1],4),periods=[x[2]['period'],x[3]['period']],review_status='pending'))
    if attachment_joins:
        collapsed=collections.defaultdict(list)
        for (c,p),ev in evidence.items():
            if find(c)!=find(p):collapsed[find(c),find(p)].extend(ev)
        evidence=collapsed;tracks={};frame_groups=[]
        for fi,f in enumerate(periods):
            for s in f['segments']:
                s['root_id']=find(s['root_id'])
                for c in s.get('previous_candidates',[]):c['previous_root_id']=find(c['previous_root_id'])
            gg=groups(f);frame_groups.append(gg)
            for rid,segments in gg.items():
                t=tracks.setdefault(rid,dict(root_id=rid,first_index=fi,observed_periods=[],total_length=0));t['observed_periods'].append(f['period']);t['total_length']+=sum(s['length_px'] for s in segments)
    ranked=[]
    for (child,p),ev in evidence.items():
        count=len({x['period'] for x in ev});score=max(x['score'] for x in ev)+.12*(count-1)
        ranked.append((score,child,p,ev))
    chosen={};relations=[]
    for score,child,p,ev in sorted(ranked,reverse=True):
        if child in chosen:continue
        alternatives=[v[0] for v in ranked if v[1]==child and v[2]!=p]
        if alternatives and score-max(alternatives)<.10:
            unresolved.append(dict(child=child,parent=p,reason='competing parent proposals',score=score));continue
        cur=p;cycle=False
        while cur in chosen:
            if cur==child:cycle=True;break
            cur=chosen[cur]
        if cur==child:cycle=True
        if cycle:unresolved.append(dict(child=child,parent=p,reason='would create parent cycle'));continue
        chosen[child]=p
        relations.append(dict(child_root_id=child,parent_root_id=p,review_status='pending',score=round(score,4),evidence=ev))
    hierarchy={}
    for rid,t in tracks.items():
        born=t['first_index'];segments=frame_groups[born][rid];counts=collections.Counter()
        for s in segments:counts.update({int(k):v for k,v in s['semantic_code_counts'].items()})
        new_fraction=counts[1]/max(1,sum(counts[k] for k in [1,2,3]))
        kind='new' if born>0 and new_fraction>=.5 else 'existing'
        hierarchy[rid]=dict(root_id=rid,label=None,parent_root_id=chosen.get(rid),family_root_id=None,
            branch_kind=kind if rid in chosen else 'independent',first_seen_period=periods[born]['period'],
            observed_periods=t['observed_periods'],review_status='pending',parentage_confirmed=False,
            first_observation_is_not_proven_birth=True,
            parent_evidence=next((v['evidence'] for v in relations if v['child_root_id']==rid),[]))
    children=collections.defaultdict(list)
    for rid,p in chosen.items():children[p].append(rid)
    def assign(rid,label,family):
        h=hierarchy[rid];h['label']=label;h['family_root_id']=family
        counters=collections.Counter()
        for c in sorted(children[rid],key=lambda c:(tracks[c]['first_index'],c)):
            kind=hierarchy[c]['branch_kind'];counters[kind]+=1;part=('N' if kind=='new' else '')+str(counters[kind]);assign(c,label+'-'+part,family)
    heads=[rid for rid in tracks if rid not in chosen]
    def family_weight(rid):
        return tracks[rid]['total_length']+sum(family_weight(c) for c in children[rid])
    heads.sort(key=lambda rid:(tracks[rid]['first_index'],-family_weight(rid),rid))
    for i,rid in enumerate(heads,1):assign(rid,f'{i:04d}',rid)
    assert all(h['label'] for h in hierarchy.values()) and len({h['label'] for h in hierarchy.values()})==len(hierarchy)
    for f in periods:
        for s in f['segments']:
            h=hierarchy[s['root_id']];s.update(display_label=h['label'],parent_root_id=h['parent_root_id'],family_root_id=h['family_root_id'],branch_kind=h['branch_kind'])
    st['root_track_hierarchy']=list(hierarchy.values());st['hierarchy_version']=1;st['parent_child_candidates']=relations
    st['unresolved_parentage']=unresolved;st['continuation_joins_v54']=joins
    st['branch_attachment_temporal_joins']=attachment_joins
    st['family_display_rule']='All descendants use the same family-root display color. A unique hierarchical label identifies every lateral. This does not replace temporal class colors.'
    st['root_tracks']=[dict(root_id=rid,first_seen_period=h['first_seen_period'],observed_periods=h['observed_periods'],display_label=h['label'],parent_root_id=h['parent_root_id'],family_root_id=h['family_root_id'],review_status='pending',biological_identity_confirmed=False) for rid,h in hierarchy.items()]
    st['cross_period_links']=[]
    for rid,h in hierarchy.items():
        observations=[(f,gg[rid]) for f,gg in zip(periods,frame_groups) if rid in gg]
        for (pf,pg),(cf,cg) in zip(observations,observations[1:]):
            st['cross_period_links'].append(dict(id=f'L-{rid}-{cf["period"]}',root_id=rid,display_label=h['label'],previous_period=pf['period'],current_period=cf['period'],previous_segment=pg[0]['id'],current_segment=cg[0]['id'],previous_segments=[s['id'] for s in pg],current_segments=[s['id'] for s in cg],kind='same_root_candidate',review_status='pending'))
    original=json.loads((OUT/'source_progress_10.json').read_text());assert doc['images']==original['images']
    for a,b in zip(before['periods'],periods):
        assert a['owner_map_rle']==b['owner_map_rle']
        assert all(x['points']==y['points'] for x,y in zip(a['segments'],b['segments']))
    (OUT/'root_structure_draft.json').write_text(json.dumps(doc,ensure_ascii=False,separators=(',',':')))
    report=dict(passed=True,original_pixel_annotation_unchanged=True,root_geometry_unchanged=True,training_modified=False,
        hierarchy_labels_unique=True,parent_graph_acyclic=True,side_labels_retained_across_dates=True,
        independent_family_roots=sum(not h['parent_root_id'] for h in hierarchy.values()),
        existing_lateral_candidates=sum(h['branch_kind']=='existing' for h in hierarchy.values()),
        new_lateral_candidates=sum(h['branch_kind']=='new' for h in hierarchy.values()),
        lateral_candidates_present_multiple_periods=sum(bool(h['parent_root_id']) and len(h['observed_periods'])>1 for h in hierarchy.values()),
        root_tracks_all_four_periods=sum(len(h['observed_periods'])==4 for h in hierarchy.values()),
        additional_continuation_joins=len(joins),unresolved_parentage_cases=len(unresolved),
        same_lateral_cross_period_attachment_joins=len(attachment_joins),
        all_structure_annotations_pending_human_review=True,
        source_SHA256=hashlib.sha256((OUT/'source_progress_10.json').read_bytes()).hexdigest(),
        examples=[h for h in hierarchy.values() if h['label'].startswith('0001-')][:5])
    (OUT/'draft_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
