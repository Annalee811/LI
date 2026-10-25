"""Propose whole-root IDs across dates, including a new root's later persistence."""
import collections,copy,hashlib,json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

ROOT=Path('/workspace/LI');OUT=ROOT/'results/B3_editor_interaction_v53_20261025'
PREV=ROOT/'results/B3_root_structure_review_v52_20261025';RUN=ROOT/'results/B3_temporal_uploaded10_v51_20261025'

def samples(segments):
    pts=[];directions=[]
    for s in segments:
        p=np.asarray(s['points'],np.float32);ii=np.arange(0,len(p),2);lo=np.maximum(0,ii-6);hi=np.minimum(len(p)-1,ii+6)
        d=p[hi]-p[lo];d/=np.maximum(np.linalg.norm(d,axis=1,keepdims=True),1e-6)
        pts.append(p[ii]);directions.append(d)
    return np.concatenate(pts),np.concatenate(directions)

def root_groups(segments):
    result={}
    for s in segments:result.setdefault(s['root_id'],[]).append(s)
    return result

def score_groups(groups,previous):
    names=list(previous);points=[];tangents=[];owners=[]
    for i,rid in enumerate(names):
        p,d=samples(previous[rid]);points.append(p);tangents.append(d);owners.append(np.full(len(p),i,np.int32))
    pp=np.concatenate(points);tt=np.concatenate(tangents);oo=np.concatenate(owners);tree=cKDTree(pp)
    result=[]
    for group in groups:
        p,d=samples(group);k=min(8,len(pp));dist,near=tree.query(p,k=k,distance_upper_bound=15)
        if k==1:dist=dist[:,None];near=near[:,None]
        valid=near<len(pp);safe=np.minimum(near,len(pp)-1);align=np.abs((d[:,None]*tt[safe]).sum(2))
        weights=np.exp(-.5*(dist/5)**2)*align**4;weights[~valid]=0
        choice=weights.argmax(1);rr=np.arange(len(p));w=weights[rr,choice];which=oo[safe[rr,choice]]
        votes=np.bincount(which,weights=w,minlength=len(names))/max(1,len(p));order=np.argsort(votes)[::-1]
        best=int(order[0]);second=float(votes[order[1]]) if len(order)>1 else 0
        good=(which==best)&(w>.25);n=int(good.sum());distance=float(np.median(dist[rr,choice][good])) if n else 99
        alignment=float(np.median(align[rr,choice][good])) if n else 0
        good_all=w>.25;dominance=n/max(1,int(good_all.sum()))
        result.append(dict(root_id=names[best],score=float(votes[best]),margin=float(votes[best])-second,
            supported_samples=n,dominance=dominance,median_distance=distance,median_alignment=alignment,
            candidates=[dict(previous_root_id=names[j],previous_segment=previous[names[j]][0]['id'],score=round(float(votes[j]),4)) for j in order[:3] if votes[j]>.02]))
    return result

def main():
    doc=json.loads((PREV/'root_structure_draft.json').read_text());st=doc['root_structure_annotation'];before=copy.deepcopy(st)
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position(st['position'])=='0353-2-B3'
    next_id=1;previous={};last_seen={};all_observations={};summaries=[]
    for fi,f in enumerate(st['periods']):
        byid={s['id']:s for s in f['segments']};parent={sid:sid for sid in byid}
        def find(sid):
            while parent[sid]!=sid:parent[sid]=parent[parent[sid]];sid=parent[sid]
            return sid
        def join(a,b):
            pa,pb=find(a),find(b)
            if pa!=pb:parent[pb]=pa;return True
            return False
        # Keep the previous draft's already grouped continuations.
        for group in root_groups(f['segments']).values():
            for s in group[1:]:join(group[0]['id'],s['id'])
        extra=0
        for node in f['nodes']:
            arms=node['arms']
            for aa,bb in node['connections']:
                a=next(x for x in arms if x['id']==aa);b=next(x for x in arms if x['id']==bb)
                if 'direction' not in a or 'direction' not in b:continue
                dot=float(np.dot(a['direction'],b['direction']))
                if dot>-.80:continue
                ca,cb=a.get('original_temporal_code',0),b.get('original_temporal_code',0)
                if ca!=cb and {ca,cb}!={2,3}:continue
                alternatives=[]
                for arm,other in [(a,b),(b,a)]:
                    alternatives.extend(float(np.dot(arm['direction'],q['direction'])) for q in arms if q['id'] not in [arm['id'],other['id']] and 'direction' in q)
                if alternatives and min(alternatives)-dot<.12:continue
                if min(byid[a['segment_id']]['length_px'],byid[b['segment_id']]['length_px'])<10:continue
                extra+=join(a['segment_id'],b['segment_id'])
        grouped={}
        for s in f['segments']:grouped.setdefault(find(s['id']),[]).append(s)
        groups=list(grouped.values());scores=score_groups(groups,previous) if previous else [None]*len(groups)
        inherited=0;newly_marked=0;unresolved=0
        for group,score in zip(groups,scores):
            counts=collections.Counter()
            for s in group:counts.update({int(k):v for k,v in s.get('semantic_code_counts',{}).items()})
            labeled=sum(counts[k] for k in [1,2,3]);new_fraction=counts[1]/max(1,labeled)
            is_new=fi>0 and new_fraction>=.65
            inherit=False
            if score and not is_new:
                inherit=(score['score']>=.27 and score['margin']>=.07 and score['supported_samples']>=5 and score['dominance']>=.65 and score['median_distance']<=7 and score['median_alignment']>=.72)
                # A long extension can have little overlap, but must retain an
                # actual old shaft with strong directional evidence.
                inherit|=(score['score']>=.12 and score['margin']>=.06 and score['supported_samples']>=20 and score['dominance']>=.85 and score['median_distance']<=5 and score['median_alignment']>=.85 and counts[3]>=8)
            if score and is_new:
                # Very strong pre-existing axis evidence is a class-review flag,
                # not a reason to invent a fresh physical identity every date.
                inherit=(score['score']>=.65 and score['margin']>=.18 and score['supported_samples']>=15 and score['dominance']>=.9 and score['median_distance']<=3 and score['median_alignment']>=.9)
            if inherit:
                rid=score['root_id'];inherited+=1;reason='prior_root_geometry_and_tangent_support'
            else:
                rid=f'R{next_id:04d}';next_id+=1
                reason='T1_baseline' if fi==0 else 'newly_observed_reference_root' if is_new else 'unresolved_previous_identity'
                newly_marked+=bool(is_new);unresolved+=bool(fi and not is_new)
            for s in group:
                s['root_id']=rid;s['review_status']='pending';s['parent_root_id']=None
                s['identity_assignment']=dict(reason=reason,score=None if not score else round(score['score'],4),provisional=True)
                s['identity_assignment']['requires_temporal_class_review']=bool(inherit and is_new)
                s['previous_candidates']=score['candidates'] if score else []
            if inherit:all_observations.setdefault(rid,[]).append(dict(period=f['period'],previous_period=last_seen.get(rid),score=round(score['score'],4)))
        current=root_groups(f['segments'])
        for node in f['nodes']:
            if node['type']!='branch' or len(node['arms'])!=3:continue
            paired={a for p in node['connections'] for a in p};main=next((a for a in node['arms'] if a['id'] in paired),None)
            if not main:continue
            parent_rid=byid[main['segment_id']]['root_id']
            for arm in node['arms']:
                if arm['id'] not in paired:
                    child=byid[arm['segment_id']]
                    if child['root_id']!=parent_rid:child['parent_root_id']=parent_rid
        for rid,group in current.items():
            previous[rid]=group;last_seen[rid]=f['period']
        # Retain last visible geometry when a root is temporarily not observed.
        summaries.append(dict(period=f['period'],trace_segments=len(f['segments']),root_groups=len(current),
            continuation_groups_inheriting_old_ID=inherited,explicit_new_groups=newly_marked,
            unresolved_old_root_groups=unresolved,additional_same_period_joins=extra))
    registry={}
    for f in st['periods']:
        for rid,group in root_groups(f['segments']).items():
            entry=registry.setdefault(rid,dict(root_id=rid,first_seen_period=f['period'],periods=[],total_length_px=0))
            entry['periods'].append(f['period']);entry['total_length_px']+=sum(s['length_px'] for s in group)
    # T1 roots visible in all four periods get the first IDs for easy review.
    ordered=sorted(registry.values(),key=lambda x:(x['first_seen_period'],-len(x['periods']),-x['total_length_px'],x['root_id']))
    rename={v['root_id']:f'R{i+1:04d}' for i,v in enumerate(ordered)}
    for f in st['periods']:
        for s in f['segments']:
            s['root_id']=rename[s['root_id']]
            if s.get('parent_root_id'):s['parent_root_id']=rename.get(s['parent_root_id'])
            for c in s.get('previous_candidates',[]):c['previous_root_id']=rename.get(c['previous_root_id'],c['previous_root_id'])
    st['root_tracks']=[];st['cross_period_links']=[]
    for entry in ordered:
        rid=rename[entry['root_id']];observed=[]
        for f in st['periods']:
            group=[s for s in f['segments'] if s['root_id']==rid]
            if group:observed.append((f,group))
        st['root_tracks'].append(dict(root_id=rid,first_seen_period=observed[0][0]['period'],observed_periods=[f['period'] for f,g in observed],
            review_status='pending',biological_identity_confirmed=False))
        for (pf,pg),(cf,cg) in zip(observed,observed[1:]):
            st['cross_period_links'].append(dict(id=f'L-{rid}-{cf["period"]}',previous_period=pf['period'],current_period=cf['period'],
                previous_segment=pg[0]['id'],current_segment=cg[0]['id'],previous_segments=[s['id'] for s in pg],current_segments=[s['id'] for s in cg],
                root_id=rid,kind='same_root_candidate',review_status='pending',score=None))
    st['identity_policy']='Identity persists across periods and across persistent/extension classes; a newly observed lateral gets its own ID, then retains it in later periods. Unresolved matches stay provisional.'
    st['proposal_method']='V53 whole-root continuation grouping and multi-date distance/tangent association; uses current reference classes to keep new laterals distinct. No model training or pixel relabeling.'
    st['human_reviewed']=False
    original=json.loads((OUT/'source_progress_10.json').read_text());assert doc['images']==original['images']
    for a,b in zip(before['periods'],st['periods']):assert a['owner_map_rle']==b['owner_map_rle'] and all(x['points']==y['points'] for x,y in zip(a['segments'],b['segments']))
    (OUT/'root_structure_draft.json').write_text(json.dumps(doc,ensure_ascii=False,separators=(',',':')))
    all_four=[t for t in st['root_tracks'] if len(t['observed_periods'])==4]
    continuing_new=[t for t in st['root_tracks'] if t['first_seen_period']!='T1' and len(t['observed_periods'])>=2]
    report=dict(passed=True,original_pixel_annotation_unchanged=True,root_geometry_unchanged=True,
        all_structure_annotations_pending_human_review=True,training_modified=False,
        source_SHA256=hashlib.sha256((OUT/'source_progress_10.json').read_bytes()).hexdigest(),
        root_tracks_all_four_periods=len(all_four),new_first_visible_tracks_retained_later=len(continuing_new),
        root_0001=next(t for t in st['root_tracks'] if t['root_id']=='R0001'),periods=summaries,
        note='Track counts and correspondences are proposals, not independently verified root identities.')
    (OUT/'draft_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
