"""Make a reviewable, unconfirmed root-identity proposal; never alter pixel labels."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path('/workspace/LI')
OUT = ROOT / 'results/B3_root_structure_review_v52_20261025'
RUN = ROOT / 'results/B3_temporal_uploaded10_v51_20261025'
PERIODS = ['T1', 'T2', 'T3', 'T4']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def unpack(v, shape):
    return np.repeat(np.asarray(v[::2], np.uint8), np.asarray(v[1::2], np.int64)).reshape(shape)


def pack(a):
    a = a.ravel()
    starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
    return np.stack([a[starts], np.diff(np.r_[starts, len(a)])], 1).astype('int64').ravel().tolist()


def main():
    import sys
    sys.path.insert(0, str(ROOT / 'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3') == '0353-2-B3'
    source = RUN / 'data/uploaded_progress_10.json'
    original = json.loads(source.read_text())
    catalog = {e['name']: e for e in original['images']}
    structure = dict(version=1, status='automatic_proposal_pending_human_review',
        position='0353-2-B3', source_annotation_SHA256=sha(source),
        coordinate_convention='Native RGB pixels, x rightwards, y downwards, origin at top-left',
        proposal_method='Reuse fixed U-Net-mask RGB/geometry axis proposals; conservative adjacent-period assignment. No new training or temporal class editing.',
        caveats=['Root IDs and crossing/branch/tip classifications are proposals, not verified biological identities.',
            'Fragmented traces can represent the same root; merge IDs after inspecting RGB.',
            'An unmatched trace is not automatically a new root.',
            'No hidden root surface or occluded tip is invented. Image-edge endpoints are not biological tips.'],
        human_reviewed=False, periods=[], cross_period_links=[])
    previous = None
    next_root = 1
    summaries = []
    for ti, t in enumerate(PERIODS):
        e = catalog[t + '_RGB.png']
        shape = e['height'], e['width']
        mask = unpack(e['layers']['mask'], shape) == 255
        temporal = unpack(e['layers']['temporal'], shape)
        rgb_path = RUN / 'data/upper' / e['name']
        assert sha(rgb_path) == e['image_sha256']
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        cache = RUN / 'axis_cache/unet'
        meta = json.loads((cache / (t + '.json')).read_text())
        with np.load(cache / (t + '.npz')) as z:
            xy = z['xy'].copy()
            owner = z['axis_map'].astype('int32')
            paths = [z[f'path_{i}'].copy() for i in range(len(meta['endkinds']))]
        assert np.array_equal(owner > 0, mask), 'Never substitute another root geometry'
        width = ndi.distance_transform_edt(mask)
        segments = []
        for i, path in enumerate(paths, 1):
            pts = xy[path]
            values, counts = np.unique(temporal[tuple(pts.T)], return_counts=True)
            histogram = {str(int(k)): int(v) for k, v in zip(values, counts)}
            dist = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            length = float(np.minimum(dist, 5).sum())
            segments.append(dict(id=f'{t}-S{i:04d}', index=i, root_id=None,
                points=pts[:, ::-1].astype(int).tolist(), length_px=round(length, 2),
                endpoint_kinds=meta['endkinds'][i-1],
                semantic_code_counts=histogram, parent_root_id=None,
                review_status='pending', note='',
                fragment_candidate=len(path) < 30 or 'junction' in meta['endkinds'][i-1]))
        scores = None
        if previous:
            # A one-to-one proposal prevents multiple new axes inheriting one old
            # identity merely because they meet at a crossing. Merges stay editable.
            pxy, powner, psegments = previous
            tree = cKDTree(pxy)
            scores = np.zeros((len(segments), len(psegments)), np.float32)
            candidates = {}
            for i, path in enumerate(paths):
                pts = xy[path]
                dist, near = tree.query(pts, distance_upper_bound=10)
                good = np.isfinite(dist)
                if not good.any():
                    continue
                ids = powner[tuple(pxy[near[good]].T)]
                weights = np.exp(-.5 * (dist[good] / 4) ** 2)
                votes = np.bincount(ids, weights=weights, minlength=len(psegments)+1)[1:]
                scores[i] = votes / max(len(pts), 1)
                top = np.argsort(scores[i])[::-1][:3]
                candidates[i] = [dict(previous_segment=psegments[j]['id'], previous_root_id=psegments[j]['root_id'], score=round(float(scores[i,j]), 4)) for j in top if scores[i,j] > .05]
            rows, cols = linear_sum_assignment(-scores)
            accepted = {}
            for i, j in zip(rows, cols):
                best = float(scores[i,j])
                other = float(np.max(np.delete(scores[i], j))) if scores.shape[1] > 1 else 0.
                if best >= .55 and best-other >= .18 and len(paths[i]) >= 12:
                    accepted[i] = j
            for i, segment in enumerate(segments):
                segment['previous_candidates'] = candidates.get(i, [])
                if i in accepted:
                    j = accepted[i]
                    segment['root_id'] = psegments[j]['root_id']
                    structure['cross_period_links'].append(dict(id=f'L-{t}-{i+1:04d}',
                        previous_period=PERIODS[ti-1], current_period=t,
                        previous_segment=psegments[j]['id'], current_segment=segment['id'],
                        root_id=segment['root_id'], score=round(float(scores[i,j]),4),
                        review_status='pending', kind='same_root_candidate'))
        for segment in segments:
            if not segment['root_id']:
                segment['root_id'] = f'R{next_root:04d}'
                next_root += 1
        # All roots, including tiny fragments, remain represented. Nothing is
        # erased from the mask or declared a confirmed whole root.
        nodes = []
        sk = np.zeros(shape, bool)
        sk[tuple(xy.T)] = True
        for decision in meta['decisions']:
            if decision.get('degree', 0) < 3 or 'point' not in decision:
                continue
            cy, cx = map(float, decision['point'])
            x0, x1 = max(0,int(cx)-18), min(shape[1],int(cx)+19)
            y0, y1 = max(0,int(cy)-18), min(shape[0],int(cy)+19)
            gy, gx = np.mgrid[y0:y1,x0:x1]
            dd = np.hypot(gx-cx,gy-cy)
            labels, nn = ndi.label(sk[y0:y1,x0:x1] & (dd>=5) & (dd<=16), np.ones((3,3)))
            arms = []
            for k in range(1,nn+1):
                yy,xx = np.where(labels==k)
                rr = dd[yy,xx]
                if len(xx)<2 or rr.max()<11:
                    continue
                far = rr>=max(10,rr.max()-3)
                px,py = float((xx[far]+x0).mean()),float((yy[far]+y0).mean())
                ids = owner[yy+y0,xx+x0]
                sid = int(np.bincount(ids).argmax())
                v = np.array([px-cx,py-cy]);v/=max(np.linalg.norm(v),1e-6)
                cc = Counter(map(int,temporal[yy+y0,xx+x0]))
                arms.append(dict(id=f'A{len(arms)+1}', segment_id=segments[sid-1]['id'],
                    x=round(px,2),y=round(py,2),direction=v.tolist(),
                    original_temporal_code=cc.most_common(1)[0][0]))
            pairing=[];taken=set()
            options=[]
            for a in range(len(arms)):
                for b in range(a+1,len(arms)):
                    dot=float(np.dot(arms[a]['direction'],arms[b]['direction']))
                    same=arms[a]['segment_id']==arms[b]['segment_id']
                    if dot<-.45:
                        options.append((dot-(.5 if same else 0),a,b))
            for _,a,b in sorted(options):
                if a in taken or b in taken:continue
                pairing.append([arms[a]['id'],arms[b]['id']]);taken|={a,b}
            candidates = np.argwhere(mask[y0:y1,x0:x1])
            if not len(candidates):continue
            closest = candidates[np.argmin((candidates[:,0]+y0-cy)**2+(candidates[:,1]+x0-cx)**2)]
            jy,jx = int(closest[0]+y0),int(closest[1]+x0)
            sid = int(owner[jy,jx])
            if arms:
                center_code=int(temporal[jy,jx])
                supported=[a for a in arms if center_code and a['original_temporal_code']==center_code]
                allowed=supported or arms
                incident={a['segment_id'] for a in allowed}
                if segments[sid-1]['id'] not in incident:
                    winner=max(allowed,key=lambda a:next(s['length_px'] for s in segments if s['id']==a['segment_id']))
                    sid=next(s['index'] for s in segments if s['id']==winner['segment_id'])
            branch = len(arms)==3 and len(pairing)==1 and any(a['original_temporal_code']==1 and a['id'] not in {v for p in pairing for v in p} for a in arms)
            kind = 'crossing' if len(arms)>=4 and len(pairing)>=2 else 'branch' if branch else 'junction_uncertain'
            node=dict(id=f'{t}-J{len(nodes)+1:04d}',type=kind,x=jx,y=jy,
                owner_segment_id=segments[sid-1]['id'], arms=arms, connections=pairing,
                review_status='pending', note='', proposed=True,
                confidence='candidate_only', ownership_radius=2,
                source_margin=decision.get('margin'),source_unresolved=decision.get('unresolved',False))
            node['owner_basis']='Incident trace, current reference class support, then continuation length; unconfirmed proposal'
            nodes.append(node)
            if branch:
                paired={a for pair in pairing for a in pair}
                main=next((a for a in arms if a['id'] in paired),None)
                if main:
                    parent=next(s for s in segments if s['id']==main['segment_id'])['root_id']
                    for arm in arms:
                        if arm['id'] not in paired:
                            child=next(s for s in segments if s['id']==arm['segment_id'])
                            if child['root_id']!=parent:child['parent_root_id']=parent
        endpoints=[]
        for segment,path in zip(segments,paths):
            if len(path)<12:
                continue  # Keep its pixels/trace, but do not call a speck a root tip.
            for side,kind in enumerate(segment['endpoint_kinds']):
                if kind not in ['tip','edge']:continue
                y,x=map(int,xy[path[0 if side==0 else -1]])
                if any((x-n['x'])**2+(y-n['y'])**2<8**2 for n in nodes):
                    continue  # A junction-adjacent fragment end is not a biological tip.
                if any((x-a)**2+(y-b)**2<9 for a,b in endpoints):continue
                endpoints.append((x,y))
                nodes.append(dict(id=f'{t}-E{len(endpoints):04d}',type='tip' if kind=='tip' else 'boundary_endpoint',
                    x=x,y=y,owner_segment_id=segment['id'],arms=[],connections=[],
                    review_status='pending',note='',proposed=True,confidence='candidate_only',ownership_radius=0))
        frame=dict(period=t,name=e['name'],image_sha256=e['image_sha256'],width=e['width'],height=e['height'],
            segments=segments,nodes=nodes,owner_map_rle=pack(owner),human_reviewed=False,
            ownership_encoding='0 background; positive integer is segment.index; segment.root_id groups fragments. Junction owner_segment_id overrides only root pixels within ownership_radius.')
        structure['periods'].append(frame)
        summaries.append(dict(period=t,root_trace_candidates=len(segments),matched_to_previous=sum(l['current_period']==t for l in structure['cross_period_links']),
            point_candidates=dict(Counter(n['type'] for n in nodes)),root_pixel_count=int(mask.sum())))
        previous=(xy,owner,segments)
    # Join only clear proposed continuations into one editable identity. Never
    # remove a short trace, alter its pixels, or turn a semantic class into another.
    next_root=1
    structure['cross_period_links']=[]
    previous_group_data=None
    for fi,frame in enumerate(structure['periods']):
        segments=frame['segments'];byid={s['id']:s for s in segments}
        parent={s['id']:s['id'] for s in segments}
        old_root_to_segment={s['root_id']:s['id'] for s in segments}
        old_parents={s['id']:old_root_to_segment.get(s['parent_root_id']) for s in segments}
        def find(sid):
            while parent[sid]!=sid:
                parent[sid]=parent[parent[sid]];sid=parent[sid]
            return sid
        merged=0
        for node in frame['nodes']:
            if node['type'] not in ['crossing','branch','junction_uncertain']:continue
            for aa,bb in node['connections']:
                a=next(v for v in node['arms'] if v['id']==aa);b=next(v for v in node['arms'] if v['id']==bb)
                if a['segment_id']==b['segment_id']:continue
                dot=float(np.dot(a['direction'],b['direction']))
                if dot>-.88 or node.get('source_unresolved'):continue
                ca,cb=a['original_temporal_code'],b['original_temporal_code']
                if ca!=cb and {ca,cb}!={2,3}:continue
                if min(byid[a['segment_id']]['length_px'],byid[b['segment_id']]['length_px'])<12:continue
                pa,pb=find(a['segment_id']),find(b['segment_id'])
                if pa!=pb:parent[pb]=pa;merged+=1
        groups={}
        for s in segments:groups.setdefault(find(s['id']),[]).append(s)
        grouped=list(groups.values())
        points=[np.unique(np.asarray([p for s in g for p in s['points']],dtype=np.int32),axis=0) for g in grouped]
        matches={};candidates={}
        if previous_group_data:
            prev_groups,prev_points=previous_group_data
            pp=np.concatenate(prev_points);owners=np.concatenate([np.full(len(p),i,np.int32) for i,p in enumerate(prev_points)])
            tree=cKDTree(pp);score=np.zeros((len(grouped),len(prev_groups)),np.float32)
            for i,pts in enumerate(points):
                dd,nn=tree.query(pts,distance_upper_bound=10);good=np.isfinite(dd)
                if good.any():score[i]=np.bincount(owners[nn[good]],weights=np.exp(-.5*(dd[good]/4)**2),minlength=len(prev_groups))/len(pts)
                top=np.argsort(score[i])[::-1][:3]
                candidates[i]=[dict(previous_segment=prev_groups[j][0]['id'],previous_root_id=prev_groups[j][0]['root_id'],score=round(float(score[i,j]),4)) for j in top if score[i,j]>.05]
            ii,jj=linear_sum_assignment(-score)
            for i,j in zip(ii,jj):
                other=float(np.max(np.delete(score[i],j))) if score.shape[1]>1 else 0
                if score[i,j]>=.55 and score[i,j]-other>=.18 and len(points[i])>=12:matches[i]=j
        for i,g in enumerate(grouped):
            if i in matches:
                j=matches[i];rid=previous_group_data[0][j][0]['root_id']
                structure['cross_period_links'].append(dict(id=f'L-{frame["period"]}-{i+1:04d}',previous_period=PERIODS[fi-1],current_period=frame['period'],previous_segment=previous_group_data[0][j][0]['id'],current_segment=g[0]['id'],root_id=rid,score=round(float(score[i,j]),4),review_status='pending',kind='same_root_candidate',previous_segments=[s['id'] for s in previous_group_data[0][j]],current_segments=[s['id'] for s in g]))
            else:rid=f'R{next_root:04d}';next_root+=1
            for s in g:s['root_id']=rid;s['previous_candidates']=candidates.get(i,[])
        for s in segments:
            p=old_parents.get(s['id']);s['parent_root_id']=byid[p]['root_id'] if p and byid[p]['root_id']!=s['root_id'] else None
        frame['continuation_groups_proposed']=merged
        summaries[fi].update(provisional_root_groups=len(grouped),same_period_continuation_merges=merged,
            matched_to_previous=len(matches),point_candidates=dict(Counter(n['type'] for n in frame['nodes'])))
        previous_group_data=grouped,points
    result=copy.deepcopy(original)
    result['root_structure_annotation']=structure
    (OUT/'source_progress_10.json').write_bytes(source.read_bytes())
    (OUT/'root_structure_draft.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':')))
    for a,b in zip(result['images'],original['images']):assert a==b
    report=dict(passed=True,original_pixel_annotation_unchanged=True,source_SHA256=sha(source),
        source_copy_SHA256=sha(OUT/'source_progress_10.json'),root_geometry_unchanged=True,
        all_structure_annotations_pending_human_review=True,training_modified=False,
        root_IDs_are_candidates_not_confirmed_root_counts=True,periods=summaries,
        new_unique_track_IDs=next_root-1,cross_period_candidates=len(structure['cross_period_links']))
    (OUT/'draft_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
