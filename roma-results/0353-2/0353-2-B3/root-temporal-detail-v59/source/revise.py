"""Upload14: keep T1/manual anchors, review later identity and first appearance.

Native RGB/foreground/width/trace coordinates and single-pixel ownership are fixed.
Only strong geometric continuations are joined; uncertain root origins stay pending.
"""
import collections, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/workspace/LI')
OUT=ROOT/'results/B3_root_temporal_review_v59_20261026'
PREV=ROOT/'results/B3_root_parentage_v58_20261026/root_structure_draft.json'
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_parentage_v58'))
from refine import tangent, root_groups, geometry, similarity

def save(name,obj):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n');tmp.replace(p)

def decode(a):return np.repeat(np.asarray(a[::2],np.int32),a[1::2])
def encode(a):
    q=np.r_[0,np.flatnonzero(a[1:]!=a[:-1])+1,len(a)]
    return np.column_stack((a[q[:-1]],np.diff(q))).ravel().tolist()

def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
    raw=(OUT/'source_progress_14.json').read_bytes();source=json.loads(raw);d=copy.deepcopy(source)
    st=d['root_structure_annotation'];assert require_active_position(st['position'])=='0353-2-B3'
    frames=st['periods'];hier={h['root_id']:h for h in st['root_track_hierarchy']}
    old=json.loads(PREV.read_text())['root_structure_annotation'];oldh={h['root_id']:h for h in old['root_track_hierarchy']}
    protected=set();changed_h=set()
    for h in hier.values():
        v=oldh.get(h['root_id'],{})
        if any(h.get(k)!=v.get(k) for k in ['label','parent_root_id','branch_kind']) or h.get('parentage_confirmed'):changed_h.add(h['root_id'])
    for f,pf in zip(frames,old['periods']):
        by={s['id']:s for s in pf['segments']}
        for s in f['segments']:
            p=by.get(s['id'],{})
            if f['period']=='T1' or s.get('review_status') in ['modified','confirmed'] or s.get('points')!=p.get('points') or s.get('root_id')!=p.get('root_id'):
                protected.add((f['period'],s['id']))
    from family_distance import repair
    family_repair=repair(st);hier={h['root_id']:h for h in st['root_track_hierarchy']}
    family_exception={(e['period'],e['segment_id']) for e in family_repair['reassigned_segments']}
    temporal=[decode(im['layers']['temporal']) for im in d['images']]
    owners=[decode(f['owner_map_rle']) for f in frames]
    t1ids={s['root_id'] for s in frames[0]['segments']}
    manual_cuts=[{(int(x),int(y)) for z in f.get('manual_disconnects',[]) for x,y in z.get('points',[])} for f in frames]
    joins=[];ambiguous=[];tip_joins=[];lateral_splits=[];mixed_axis_splits=[];attachment_reviews=[]
    nextroot=max(int(r[1:]) for r in hier)+1
    provisional_number=max([int(h['label']) for h in hier.values() if h['label'].isdigit()]+[0])+1
    # Registered same-axis overlap, never simply nearest root/connected foreground.
    # T1 fixes identity; subsequent manual edits remain untouched.
    for fi in [1,2,3]:
        f=frames[fi];prior=[{r:geometry(ss) for r,ss in root_groups(pf).items()} for pf in frames[:fi]]
        for s in sorted(f['segments'],key=lambda s:-s['length_px']):
            if (f['period'],s['id']) in protected or len(s['points'])<18 or s['length_px']<22:continue
            current=s['root_id'];p=np.asarray(s['points']);existing=max([similarity(s,g[current])[0] for g in prior if current in g]+[0])
            if existing>=.7:continue
            rank=[]
            for j,g in enumerate(prior):
                for r,geom in g.items():
                    if r==current:continue
                    if (geom['p'].max(0)+4<p.min(0)).any() or (geom['p'].min(0)-4>p.max(0)).any():continue
                    cov,dist,ang,score=similarity(s,geom)
                    if cov>=.9 and dist<=3.5 and ang>=.97:
                        rank.append(dict(root_id=r,period='T'+str(j+1),coverage=cov,distance=dist,alignment=ang,score=score))
            byroot=collections.defaultdict(list)
            for q in rank:byroot[q['root_id']].append(q)
            ranked=sorted([(max(x['score'] for x in ev),r,ev) for r,ev in byroot.items()],reverse=True)
            if not ranked:continue
            score,r,ev=ranked[0]
            # A current hand-set parent may not be replaced by a different family.
            family=lambda a:hier[a].get('family_root_id',a)
            eligible=existing<.3 and (len(ev)>=2 or any(x['period']=='T1' for x in ev))
            repaired_roots={q['root_id'] for q in tip_joins+mixed_axis_splits+lateral_splits if int(q['period'][1:])<fi+1}
            if not eligible and existing<.2 and r in repaired_roots and max(q['coverage'] for q in ev)>.95 and s['length_px']>20:
                eligible=True
                event_basis='same axis as an earlier accepted through-continuation; propagate repaired identity'
            else:event_basis='registered same-axis overlap to earlier root; not a new birth'
            if not eligible and existing<.15 and len(ev)==1 and ev[0]['coverage']>.95 and ev[0]['alignment']>.98 and s['length_px']>60:
                # Newly visible long axes must continue from T2 even when T1
                # offers only an unrelated tiny boundary fragment with this ID.
                j=int(ev[0]['period'][1:])-1;seed=root_groups(frames[j])[r];seedpoints=np.concatenate([q['points'] for q in seed]);new_fraction=float(np.mean(temporal[j][seedpoints[:,1]*frames[j]['width']+seedpoints[:,0]]==1));prior_current_length=max([sum(q['length_px'] for q in root_groups(pf).get(current,[])) for pf in frames[:fi]]+[0])
                eligible=new_fraction>=.85 and sum(q['length_px'] for q in seed)>=100 and prior_current_length<25
            eligible &= current not in changed_h or family(current)==family(r)
            eligible &= len(ranked)==1 or score-ranked[1][0]>=.18
            event=dict(period=f['period'],segment_id=s['id'],previous_root_id=current,root_id=r,point=s['points'][len(p)//2],prior_current_support=round(existing,4),evidence=ev)
            if not eligible:event['reason']='manual parent/current-axis/alternative ambiguity retained';ambiguous.append(event);continue
            s['root_id']=r;s['review_status']='pending';event['reason']=event_basis;joins.append(event)
        # A foreground skeleton is cut into pieces at crossings. Connect only
        # opposed terminal tangents supported by an EARLIER same-root axis;
        # adjacency or a root's existing ID alone is insufficient.
        for iteration in range(2):
            groups=root_groups(f);ends=[];plans=[]
            for r,ss in groups.items():
                for body in ss:
                    if len(body['points'])<12:continue
                    evidence=[similarity(body,g[r]) for g in prior if r in g]
                    if not evidence or max(q[0] for q in evidence)<.7:continue
                    bp=np.asarray(body['points'])
                    for end in [0,-1]:
                        b=bp[end];vec=b-(bp[8] if end==0 else bp[-9]);vec=vec/max(np.linalg.norm(vec),1e-9)
                        ends.append((r,body,b,vec))
            if not ends:break
            endtree=cKDTree([e[2] for e in ends])
            for s in f['segments']:
                if (f['period'],s['id']) in protected or len(s['points'])<15:continue
                current=s['root_id'];evidence=[similarity(s,g[current]) for g in prior if current in g]
                if evidence and max(q[0] for q in evidence)>.3:continue
                sp=np.asarray(s['points']);options=[]
                for end in [0,-1]:
                    ep=sp[end];vec=ep-(sp[8] if end==0 else sp[-9]);vec=vec/max(np.linalg.norm(vec),1e-9)
                    for ix in endtree.query_ball_point(ep,5):
                        r,body,bp,bv=ends[ix]
                        if r==current:continue
                        align=float(-vec@bv)
                        if align<.97:continue
                        support=[similarity(s,g[r]) for g in prior if r in g]
                        if not support or max(q[0] for q in support)<.3:continue
                        best=max(support,key=lambda q:q[0])
                        if best[2]<.97:continue
                        # Maintain manual family correction as an anchor.
                        if current in changed_h and hier[current]['family_root_id']!=hier[r]['family_root_id']:continue
                        options.append((align-float(np.linalg.norm(ep-bp))*.01,r,body['id'],align,float(np.linalg.norm(ep-bp)),best[0]))
                options.sort(reverse=True)
                if not options or len(options)>1 and options[0][1]!=options[1][1] and options[0][0]-options[1][0]<.1:continue
                _,r,body,align,dist,support=options[0];plans.append((s,r,body,align,dist,support))
            for s,r,body,align,dist,support in plans:
                before=s['root_id'];s['root_id']=r;s['review_status']='pending';tip_joins.append(dict(period=f['period'],segment_id=s['id'],previous_root_id=before,root_id=r,body_segment=body,point=s['points'][0],opposed_tangent_alignment=round(align,4),endpoint_distance=round(dist,3),previous_same_axis_coverage=round(support,4),reason='same-axis through-continuation at a skeleton cut; no new identity'))
        # A new side attached to a continuing through-mother must not share its
        # mother's ID. Only an unedited new-class one-sided branch is eligible.
        groups=root_groups(f)
        for r,ss in list(groups.items()):
            if not any(r in g for g in prior) or len(ss)<2:continue
            for s in ss:
                if (f['period'],s['id']) in protected or s['length_px']<30:continue
                sp=np.asarray(s['points']);v=temporal[fi][sp[:,1]*f['width']+sp[:,0]]
                if np.mean(v==1)<.9:continue
                oldmatch=max([similarity(s,g[r])[0] for g in prior if r in g]+[0])
                if oldmatch>.15:continue
                body=[q for q in ss if q['id']!=s['id'] and max([similarity(q,g[r])[0] for g in prior if r in g]+[0])>.7]
                if not body:continue
                bg=geometry(body);options=[]
                for end in [0,-1]:
                    ep=sp[end];dist,ix=bg['tree'].query(ep)
                    if dist>12:continue
                    cdir=(sp[8]-ep) if end==0 else (sp[-9]-ep);cdir=cdir/max(np.linalg.norm(cdir),1e-9);dot=abs(float(cdir@bg['t'][ix]))
                    if dot>.88:continue
                    origin=bg['p'][ix];local=bg['p'][np.linalg.norm(bg['p']-origin,axis=1)<30];proj=(local-origin)@bg['t'][ix]
                    if np.count_nonzero(proj<-8)<6 or np.count_nonzero(proj>8)<6:continue
                    childlocal=sp[np.linalg.norm(sp-ep,axis=1)<35]
                    if np.count_nonzero((childlocal-ep)@cdir<-7)>3:continue
                    options.append((dist,end,ep.tolist(),dot))
                if len(options)!=1:continue
                dist,end,point,dot=options[0];newr='R'+str(nextroot).zfill(4);nextroot+=1
                certain_parent=dist<=6 and not hier[r].get('parent_root_id')
                if not certain_parent:
                    # Distinct new axis is real in the current annotation, but
                    # proximity/crossing or inherited deep hierarchy cannot
                    # prove its mother. Separate identity, leave role pending.
                    label=str(provisional_number).zfill(4);provisional_number+=1
                    hier[newr]=dict(root_id=newr,label=label,parent_root_id=None,family_root_id=newr,branch_kind='independent',review_status='pending',parentage_confirmed=False,birth_role_v58='unresolved_side_vs_independent')
                    s['root_id']=newr;s['review_status']='pending';record=dict(period=f['period'],segment_id=s['id'],previous_shared_root_id=r,root_id=newr,label=label,point=point,candidate_mother_root_id=r,endpoint_distance_px=round(float(dist),3),axis_dot=round(dot,4),reason='different new-class axis cannot share the old root ID; exact mother versus crossing unresolved')
                    mixed_axis_splits.append(record);attachment_reviews.append(record);continue
                reserved=[int(h['label'].split('-')[-1][1:]) for h in hier.values() if h.get('parent_root_id')==r and h['label'].split('-')[-1].startswith('N') and h['label'].split('-')[-1][1:].isdigit()]
                label=hier[r]['label']+'-N'+str(max(reserved+[0])+1)
                hier[newr]=dict(root_id=newr,label=label,parent_root_id=r,family_root_id=hier[r]['family_root_id'],branch_kind='new',review_status='pending',parentage_confirmed=False,birth_role_v58='new_lateral')
                s['root_id']=newr;s['review_status']='pending';lateral_splits.append(dict(period=f['period'],segment_id=s['id'],previous_shared_mother_root_id=r,root_id=newr,label=label,point=point,mother_distance_px=round(float(dist),3),axis_dot=round(dot,4),new_class_fraction=round(float(np.mean(v==1)),4),reason='one-sided new branch on a continuing mother axis; distinct side identity'))
        print('identity checked',f['period'],len(joins),'through-joins',len(tip_joins),'new-side-splits',len(lateral_splits),flush=True)
    grouped=[root_groups(f) for f in frames]
    present=collections.defaultdict(list)
    for fi,g in enumerate(grouped):
        for r in g:present[r].append(fi)
    geom=[{r:geometry(ss) for r,ss in g.items()} for g in grouped]
    # Audit first-visible axes against ALL earlier roots and plausible attachment
    # locations. An interior origin is allowed: boundary entry is not required.
    birth=[];renames=[];parent_proposals=[];used={h['label'] for h in hier.values()}
    historical=[label for h in hier.values() for label in [h['label']]+h.get('previous_display_labels_v57',[]) if label and label.startswith('N-') and label[2:].isdigit()]
    counter=max([int(x[2:]) for x in historical]+[0])+1
    for r,dates in sorted(present.items()):
        h=hier[r];first=dates[0]
        if first==0:continue
        ss=grouped[first][r];p=np.concatenate([s['points'] for s in ss]);length=sum(s['length_px'] for s in ss)
        values=temporal[first][p[:,1]*frames[first]['width']+p[:,0]];new=float(np.mean(values==1));ext=float(np.mean(values==2));prior=[]
        for j in range(first):
            scores=[]
            for oldr,g in geom[j].items():
                if (g['p'].max(0)+6<p.min(0)).any() or (g['p'].min(0)-6>p.max(0)).any():continue
                cv,di,an,sc=similarity(dict(points=p),g);scores.append((cv,sc,oldr,di,an))
            best=max(scores,default=(0,0,None,999,0));prior.append(dict(period='T'+str(j+1),root_id=best[2],coverage=round(best[0],4),distance=round(best[3],3),alignment=round(best[4],4)))
        possible=[]
        for s in ss:
            sp=np.asarray(s['points']);direction=tangent(sp)
            for end in [0,-1]:
                ep=sp[end]
                for other,g in geom[first].items():
                    if other==r:continue
                    dist,idx=g['tree'].query(ep)
                    if dist>8:continue
                    # Near-parallel roots crossing/aligning do not establish a mother.
                    angle=abs(float(np.dot(direction[end],g['t'][idx])))
                    if angle>.93:continue
                    point=g['p'][idx];axis=g['t'][idx];local=g['p'][np.linalg.norm(g['p']-point,axis=1)<25];proj=(local-point)@axis
                    if np.count_nonzero(proj<-7)<5 or np.count_nonzero(proj>7)<5:continue
                    # A whole child continuing on both sides is crossing, not origin.
                    childdir=(sp[min(8,len(sp)-1)]-ep) if end==0 else (sp[max(0,len(sp)-9)]-ep);childdir=childdir/max(np.linalg.norm(childdir),1e-9)
                    childlocal=p[np.linalg.norm(p-ep,axis=1)<35];back=int(np.count_nonzero((childlocal-ep)@childdir<-7))
                    if back>3:continue
                    possible.append(dict(parent_root_id=other,point=ep.tolist(),distance=round(float(dist),3),axis_dot=round(angle,4)))
        parent=h.get('parent_root_id');confirmed=bool(h.get('independent_new_confirmed_v58'))
        prior_max=max([e['coverage'] for e in prior]+[0]);supported=not parent and new>=.85 and prior_max<.15 and not possible and ext<.1 and length>=25
        # Small native fragments may receive a candidate N-number, never a
        # false statement of confirmed independent biology.
        role='new_lateral' if parent and h.get('branch_kind')=='new' else 'existing_lateral' if parent else 'independent_new_confirmed' if confirmed else 'independent_new_supported' if supported else 'unresolved_side_vs_independent'
        if supported and not h['label'].startswith('N-') and r not in t1ids:
            oldlabel=h['label'];label='N-'+str(counter).zfill(4);counter+=1
            while label in used:label='N-'+str(counter).zfill(4);counter+=1
            h['label']=label;used.add(label);h['previous_display_labels_v57']=list(dict.fromkeys(h.get('previous_display_labels_v57',[])+[oldlabel]));renames.append(dict(root_id=r,old_label=oldlabel,label=label,first_seen_period='T'+str(first+1),axis_length_px=round(length,3),candidate_only=True))
        h['birth_role_v58']=role;h['independent_new_evidence_v59']=dict(first_observation='T'+str(first+1),new_class_fraction=round(new,4),extension_class_fraction=round(ext,4),prior_axis_matches=prior,plausible_mother_attachments=possible,confirmed_by_user=confirmed,boundary_entry_not_required=True)
        rec=dict(root_id=r,display_label=h['label'],first_seen_period='T'+str(first+1),role=role,axis_length_px=round(length,3),point=p[len(p)//2].tolist(),**h['independent_new_evidence_v59']);birth.append(rec)
        if possible and not parent:parent_proposals.append(rec)
    # Cascade labels to descendants, but never change an identity present in T1.
    children=collections.defaultdict(list)
    for r,h in hier.items():children[h.get('parent_root_id')].append(r)
    cascade=[]
    def visit(r,family):
        h=hier[r];h['family_root_id']=family
        for c in children[r]:
            ch=hier[c];suffix=ch['label'].split('-')[-1];label=h['label']+'-'+suffix
            if label!=ch['label']:
                assert c not in t1ids,'No retrospective T1 relabel'
                oldlabel=ch['label'];ch['label']=label;cascade.append(dict(root_id=c,old_label=oldlabel,label=label))
            visit(c,family)
    for r in children[None]:visit(r,r)
    # Class corrections need actual earlier same-identity geometry, not just an
    # ID's registry. Change only new/untyped pixels on near-exact pre-existing
    # segments; preserve extension and ambiguous crossings. Human-modified
    # segments are protected. A later growing terminal keeps original class2.
    class_changes=[];class_reviews=[]
    for fi in [1,2,3]:
        f=frames[fi];before=temporal[fi].copy()
        for s in f['segments']:
            r=s['root_id'];p=np.asarray(s['points']);v=temporal[fi][p[:,1]*f['width']+p[:,0]];newfrac=float(np.mean(v==1));pre=[geom[j][r] for j in range(fi) if r in geom[j]]
            if not pre or newfrac<.15:continue
            check=[similarity(s,g) for g in pre];best=max(check,key=lambda a:a[0]);cv,dist,ang,_=best
            if cv<.9 or dist>2.5 or ang<.97:continue
            record=dict(period=f['period'],segment_id=s['id'],root_id=r,label=hier[r]['label'],point=s['points'][len(p)//2],new_fraction=round(newfrac,4),prior_same_axis_coverage=round(cv,4),distance=round(dist,3),alignment=round(ang,4))
            if (f['period'],s['id']) in protected:record['reason']='manual edit retained; temporal conflict flagged';class_reviews.append(record);continue
            pixel=np.flatnonzero((owners[fi]==s['index']) & np.isin(temporal[fi],[0,1]));yx=np.c_[pixel%f['width'],pixel//f['width']];trees=[g['tree'] for g in pre]
            if len(pixel):
                ds=np.min(np.stack([tree.query(yx)[0] for tree in trees]),axis=0);pixel=pixel[ds<5]
            if len(pixel):temporal[fi][pixel]=3;record['pixels_new_to_persistent']=len(pixel);class_changes.append(record)
        if not np.array_equal(before,temporal[fi]):d['images'][fi]['layers']['temporal']=encode(temporal[fi])
    # All proposals are explicit. Crossings retain the original unique owner.
    # Segment metadata is synchronized with authoritative native class pixels.
    gaps=[];lifetime=[]
    for r,h in hier.items():
        if r not in present:continue
        dates=present[r];h['first_seen_period']='T'+str(dates[0]+1);h['observed_periods']=['T'+str(j+1) for j in dates]
        if dates[0]==0:h['birth_role_v58']='existing_lateral' if h.get('parent_root_id') else 'baseline_independent'
        states=[]
        for j in range(4):
            status='before_first_observation' if j<dates[0] else 'observed_same_identity' if j in dates else 'observation_gap_identity_retained' if any(q>j for q in dates) else 'not_reliably_matched_identity_retained'
            states.append(dict(period='T'+str(j+1),status=status,root_id=r,label=h['label'],parent_root_id=h.get('parent_root_id'),observed=j in dates))
            if j>=dates[0] and j not in dates:gaps.append(dict(root_id=r,display_label=h['label'],period='T'+str(j+1),status=status,point=grouped[dates[0]][r][0]['points'][0]))
        lifetime.append(dict(root_id=r,label=h['label'],parent_root_id=h.get('parent_root_id'),birth_role=h.get('birth_role_v58'),states=states,identity_retained_through_T4=True,visibility_is_not_identity=True))
    for fi,f in enumerate(frames):
        if fi==0:continue
        for s in f['segments']:
            h=hier[s['root_id']];s.update(display_label=h['label'],parent_root_id=h.get('parent_root_id'),family_root_id=h['family_root_id'],branch_kind=h['branch_kind']);p=np.asarray(s['points']);v=temporal[fi][p[:,1]*f['width']+p[:,0]];codes,counts=np.unique(v,return_counts=True);s['semantic_code_counts']={str(int(k)):int(v) for k,v in zip(codes,counts)}
    st['root_track_hierarchy']=list(hier.values());st['root_lifetime_registry_v58']=lifetime
    st['root_tracks']=[dict(root_id=t['root_id'],display_label=t['label'],parent_root_id=t['parent_root_id'],first_seen_period=hier[t['root_id']]['first_seen_period'],observed_periods=hier[t['root_id']]['observed_periods'],family_root_id=hier[t['root_id']]['family_root_id'],branch_kind=hier[t['root_id']]['branch_kind'],review_status='pending',biological_identity_confirmed=False) for t in lifetime]
    st['cross_period_links']=[]
    for r,dates in present.items():
        for a,b in zip(dates,dates[1:]):
            ga,gb=grouped[a][r],grouped[b][r];st['cross_period_links'].append(dict(id='L-'+r+'-T'+str(b+1),previous_period='T'+str(a+1),current_period='T'+str(b+1),root_id=r,display_label=hier[r]['label'],previous_segment=ga[0]['id'],current_segment=gb[0]['id'],previous_segments=[s['id'] for s in ga],current_segments=[s['id'] for s in gb],kind='same_root_candidate',review_status='pending'))
    # Drop only obsolete *automatic* pairs after identity joins, never edited ones.
    pair_conflicts=[]
    for f in frames[1:]:
        by={s['id']:s for s in f['segments']}
        for n in f['nodes']:
            keep=[]
            for pair in n['connections']:
                rr=[by[next(a['segment_id'] for a in n['arms'] if a['id']==arm)]['root_id'] for arm in pair]
                manual=n.get('review_status') in ['modified','confirmed']
                if rr[0]==rr[1] or manual:keep.append(pair)
                else:pair_conflicts.append(dict(period=f['period'],node_id=n['id'],pair=pair,root_ids=rr))
            n['connections']=keep
    audit=dict(source_upload='root_annotation_progress (14).json',source_SHA256=hashlib.sha256(raw).hexdigest(),T1_unchanged=False,T1_changes_explicitly_requested_in_screenshot=family_repair,foreground_width_length_and_owner_maps_unchanged=True,manual_segments_protected=len(protected),manual_hierarchy_roots=sorted(changed_h),continuation_recoveries=joins,through_continuation_joins=tip_joins,new_lateral_identity_splits=lateral_splits,mixed_new_axes_separated_pending_parentage=mixed_axis_splits,uncertain_attachment_reviews=attachment_reviews,uncertain_continuations=ambiguous,new_independent_numbering=renames,descendant_label_updates=cascade,birth_reviews=birth,parentage_reviews=parent_proposals,temporal_pixel_corrections=class_changes,manual_temporal_conflicts=class_reviews,visibility_gaps=gaps,obsolete_automatic_connections_removed=pair_conflicts,training_started=False)
    st['temporal_revision_v59']=audit;st['human_reviewed']=False;st['status']='proposal_under_review'
    # Existing V58 review queues are kept as history, not mistaken for V59 edits.
    st['proposal_method']='Upload14 anchors; T1 fixed; earlier whole-axis continuation; independent-new prefix after lateral/continuation screening; proposal for review'
    assert d['images'][0]==source['images'][0]
    assert {k:v for k,v in frames[0].items() if k!='segments'}=={k:v for k,v in source['root_structure_annotation']['periods'][0].items() if k!='segments'}
    assert len({h['label'] for h in hier.values()})==len(hier)
    for fi,(f,src) in enumerate(zip(frames,source['root_structure_annotation']['periods'])):
        assert f['owner_map_rle']==src['owner_map_rle'];assert [(s['id'],s['points']) for s in f['segments']]==[(s['id'],s['points']) for s in src['segments']]
        for k,v in source['images'][fi]['layers'].items():
            if k!='temporal':assert d['images'][fi]['layers'][k]==v
        for s,oldseg in zip(f['segments'],src['segments']):
            if (f['period'],s['id']) in protected and (f['period'],s['id']) not in family_exception:assert s['root_id']==oldseg['root_id']
    save('root_structure_draft.json',d);save('revision_audit.json',audit)
    stats=dict(passed=True,source_SHA256=audit['source_SHA256'],T1_pixels_geometry_nodes_and_ownership_exact=True,T1_targeted_family_identity_repair=True,screenshot_source_family_label='0030',separated_T1_lower_root_label=family_repair['lower_label'],family_root_segment_reassignments=len(family_repair['reassigned_segments']),family_parent_changes=len(family_repair['parent_changes']),all_foreground_pixels_and_trace_geometry_exact=True,all_single_owner_maps_exact=True,manual_parentage_preserved_outside_requested_family=True,continuation_joins=len(joins),through_continuation_joins=len(tip_joins),new_lateral_identity_splits=len(lateral_splits),mixed_new_axes_separated_pending_parentage=len(mixed_axis_splits),N_prefix_assignments=len(renames),N_first_seen_by_period=dict(collections.Counter(x['first_seen_period'] for x in renames)),temporal_changed_segments=len(class_changes),temporal_changed_pixels=sum(x['pixels_new_to_persistent'] for x in class_changes),manual_temporal_conflicts=len(class_reviews),identity_gaps=len(gaps),final_json_SHA256=hashlib.sha256((OUT/'root_structure_draft.json').read_bytes()).hexdigest())
    save('draft_verification.json',stats);print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
