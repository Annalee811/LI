"""Screenshot-directed curation of disconnected upload14 family0030.

The screenshot's older 0005 display label is not used as a lookup key: native
RGB coordinates identify R0151 in upload14. Preserve its upper axis; split its
distant lower axis and unrelated residual components without drawing new pixels.
"""
import collections, copy
import numpy as np
from scipy.spatial import cKDTree
from refine import geometry, similarity, tangent

def repair(st):
    frames=st['periods'];hier={h['root_id']:h for h in st['root_track_hierarchy']};target='R0151'
    assert hier[target]['label']=='0030'
    original_h=copy.deepcopy(hier);counter=max(int(r[1:]) for r in hier)+1
    labelnum=max([int(h['label']) for h in hier.values() if h['label'].isdigit()]+[0])+1
    records=[];reparents=[];components=[];allowed=set();original_parent={r:h.get('parent_root_id') for r,h in hier.items()}
    def newroot(reason):
        nonlocal counter,labelnum
        rid='R'+str(counter).zfill(4);counter+=1;label=str(labelnum).zfill(4);labelnum+=1
        hier[rid]=dict(root_id=rid,label=label,parent_root_id=None,family_root_id=rid,branch_kind='independent',parentage_confirmed=False,review_status='pending',birth_role_v58='unresolved_side_vs_independent',identity_revision_reason_v59=reason)
        return rid
    lower=newroot('T1-existing lower axis has no supported connection to the distant upper R0151 axis; not a later N-root')
    anchors=[('T1-S0005','T1-S0199'),('T2-S0006','T2-S0727'),('T3-S0007','T3-S0713'),('T4-S0006','T4-S0293')]
    previous=[]
    for fi,f in enumerate(frames):
        ss=[s for s in f['segments'] if s['root_id']==target];by={s['id']:i for i,s in enumerate(ss)};parent=list(range(len(ss)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        def union(i,j):parent[find(j)]=find(i)
        for i,a in enumerate(ss):
            ap=np.asarray(a['points']);aa=tangent(ap)
            for j,b in enumerate(ss[:i]):
                bp=np.asarray(b['points']);bb=tangent(bp);connected=False
                if len(ap)<5 or len(bp)<5:
                    # Only an actually co-located residual may join an axis.
                    connected=min(cKDTree(ap).query(bp)[0])<3
                else:
                    for ae in [0,-1]:
                        av=ap[ae]-(ap[min(7,len(ap)-1)] if ae==0 else ap[max(0,len(ap)-8)]);av=av/max(np.linalg.norm(av),1e-9)
                        for be in [0,-1]:
                            bv=bp[be]-(bp[min(7,len(bp)-1)] if be==0 else bp[max(0,len(bp)-8)]);bv=bv/max(np.linalg.norm(bv),1e-9)
                            if np.linalg.norm(ap[ae]-bp[be])<=10 and -av@bv>.90:connected=True
                    if not connected:
                        for trace,body in [(a,b),(b,a)]:
                            cv,di,an,score=similarity(trace,geometry([body]))
                            if cv>.85 and di<3 and an>.94:connected=True
                if connected:union(i,j)
        groups=collections.defaultdict(list)
        for i,s in enumerate(ss):groups[find(i)].append(s)
        top_index=find(by[anchors[fi][0]]);low_index=find(by[anchors[fi][1]])
        assert top_index!=low_index,'A claimed separating gap must be real'
        dated=[]
        for key,segments in groups.items():
            points=np.concatenate([s['points'] for s in segments]);length=sum(s['length_px'] for s in segments)
            if key==top_index:rid=target;basis='explicit screenshot upper-axis anchor, native RGB checked'
            elif key==low_index:rid=lower;basis='explicit lower-axis anchor; follows same visible lower axis in all four dates'
            else:
                candidates=[]
                for pf,r,ps in previous:
                    pg=geometry(ps)
                    if (pg['p'].max(0)+7<points.min(0)).any() or (pg['p'].min(0)-7>points.max(0)).any():continue
                    if length>=12:
                        cv,di,an,score=similarity(dict(points=points),pg)
                        if cv>.8 and di<5 and an>.93:candidates.append((score,r))
                    elif len(points)<=6:
                        distance=float(np.median(pg['tree'].query(points)[0]))
                        if distance<3:candidates.append((1-distance*.15,r))
                candidates.sort(reverse=True)
                if candidates and (len(candidates)==1 or candidates[0][1]==candidates[1][1] or candidates[0][0]-candidates[1][0]>.15):rid=candidates[0][1];basis='same-axis residual component matched to an earlier date'
                else:rid=newroot('Disconnected former R0151 component; role and mother unconfirmed');basis='unrelated/disconnected component cannot inherit the distant mother; provisional identity'
            for s in segments:
                if rid!=target:
                    allowed.add((f['period'],s['id']));s['root_id']=rid;s['review_status']='pending';records.append(dict(period=f['period'],segment_id=s['id'],previous_root_id=target,root_id=rid,point=s['points'][len(s['points'])//2],basis=basis))
            dated.append((fi,rid,segments));components.append(dict(period=f['period'],root_id=rid,segment_ids=[s['id'] for s in segments],length_px=round(length,3),bbox=[points.min(0).tolist(),points.max(0).tolist()],basis=basis))
        previous.extend(dated)
    # Parentage is local. Child axes with no close contact to either the upper
    # or lower main axis are not forced into either family merely by old names.
    for r,h in list(hier.items()):
        if original_parent.get(r)!=target:continue
        matches=[]
        for fi,f in enumerate(frames):
            child=[s for s in f['segments'] if s['root_id']==r]
            if not child:continue
            ends=np.asarray([q for s in child for q in [s['points'][0],s['points'][-1]]])
            for main in [target,lower]:
                body=[s for s in f['segments'] if s['root_id']==main]
                if not body:continue
                distance=float(cKDTree(np.concatenate([s['points'] for s in body])).query(ends)[0].min())
                if distance<=12:matches.append(dict(period=f['period'],parent_root_id=main,distance_px=round(distance,3)))
        top=[e for e in matches if e['parent_root_id']==target];bottom=[e for e in matches if e['parent_root_id']==lower]
        if bottom and not top:newparent=lower;reason='child endpoint attaches near the lower visible main axis, never the remote upper axis'
        elif top and not bottom:continue
        else:newparent=None;reason='no unique supported endpoint contact to either visible main; old remote parentage withdrawn for review'
        h['parent_root_id']=newparent;h['parentage_confirmed']=False;h['review_status']='pending'
        if newparent is None:h['label']=str(labelnum).zfill(4);labelnum+=1;h['branch_kind']='independent'
        reparents.append(dict(root_id=r,previous_parent=target,parent_root_id=newparent,reason=reason,evidence=matches))
    children=collections.defaultdict(list)
    for r,h in hier.items():children[h.get('parent_root_id')].append(r)
    renamed=[]
    def visit(r,family):
        hier[r]['family_root_id']=family
        for c in children[r]:
            h=hier[c];old=h['label'];label=hier[r]['label']+'-'+old.split('-')[-1]
            if label!=old:h['label']=label;h['previous_display_labels_v57']=list(dict.fromkeys(h.get('previous_display_labels_v57',[])+[old]));renamed.append(dict(root_id=c,previous_label=old,label=label))
            visit(c,family)
    for r in children[None]:visit(r,r)
    for f in frames:
        for s in f['segments']:
            h=hier[s['root_id']]
            if s['root_id'] in {x['root_id'] for x in renamed} or (f['period'],s['id']) in allowed:s.update(display_label=h['label'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind'])
    st['root_track_hierarchy']=list(hier.values())
    return dict(screenshot_older_label='0005',upload14_actual_label='0030',original_internal_root_id=target,upper_root_id=target,lower_root_id=lower,lower_label=hier[lower]['label'],screenshot_native_coordinate_match=True,source_parentage_is_not_spatial_proof=True,reassigned_segments=records,parent_changes=reparents,renames=renamed,components=components,lower_is_T1_existing_not_N=True,automatic_changes_pending_human_review=True)
