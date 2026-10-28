"""Freeze and audit real prediction repairs, including all user-reported examples."""
import csv
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path('/workspace/LI'); HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_root_state_prediction_repair_v88_20261028'
PREV=ROOT/'results/B3_branch_continuity_review_v85_20261028'
REF=ROOT/'results/B3_T4_all_extension_reference_to_new_v87_20261028'
DATA=ROOT/'results/B3_original_UNet_latestCG_upload24_v82_20261028/data'
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
PERIODS=['T1','T2','T3','T4']
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from prepare import js,csvout,sha
from evaluate import metric
from rendering import Canvas,isolated,overlay


def readcsv(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))


def main():
    verify=json.loads((OUT/'prediction_verification.json').read_text()); assert verify['passed']
    for t,h in verify['prediction_sha256'].items(): assert sha(OUT/'full'/(t+'_classes.png'))==h
    js(OUT/'prediction_freeze.json',dict(prediction_sha256=verify['prediction_sha256'],no_model_training=True,no_manual_reference_in_prediction=True,freeze_before_metrics=True,feedback_scope='All examples are known user-feedback development cases; the old internal spatial test mask is now an audit region, not independent validation.'))
    with (OUT/'root_axis_snapshots.pkl').open('rb') as f: snapshots=pickle.load(f)
    before={t:np.array(Image.open(PREV/'full'/(t+'_classes.png'))) for t in PERIODS}
    after={t:np.array(Image.open(OUT/'full'/(t+'_classes.png'))) for t in PERIODS}
    gt={t:np.array(Image.open(REF/'corrected_reference'/(t+'_classes.png'))) for t in PERIODS}
    rgb={t:np.array(Image.open(BASE/'data/upper'/(t+'_RGB.png'))) for t in PERIODS}
    spatial=np.load(DATA/'spatial.npy'); rows=[]; benefits=[]; results={}; pixrows=[]; identityrows=[]
    for t in PERIODS:
        assert np.array_equal(before[t]>0,after[t]>0)
        if t=='T1': assert np.array_equal(before[t],after[t])
        edit=before[t]!=after[t]
        pixrows.append(dict(period=t,changed_pixels=int(edit.sum()),foreground_added=0,foreground_removed=0))
        owner=np.load(OUT/'ownership'/(t+'_axis_owner.npz'))['axis']
        for r in snapshots[t]['states']:
            if r['state'] not in ['new','established']: continue
            use=owner==r['axis']; old=before[t][use]; new=after[t][use]
            forbid=lambda v:int((v!=2).sum()) if r['state']=='new' else int((v==2).sum())
            identityrows.append(dict(period=t,axis=r['axis'],state=r['state'],owned_pixels=int(use.sum()),forbidden_before=forbid(old),forbidden_after=forbid(new)))
    assert not any(r['forbidden_after'] for r in identityrows)
    csvout(OUT/'Native_pixel_change_audit.csv',pixrows);csvout(OUT/'Accepted_root_state_consistency.csv',identityrows)
    for scope in ['original_inner_spatial_audit','all_valid_annotated_root_pixels']:
        yy=[];pp={name:[] for name in ['previous_V85','root_axis_V88']}
        for t in PERIODS[1:]:
            take=np.isin(gt[t],[1,2,3])&(before[t]>0)
            if scope=='original_inner_spatial_audit': take&=spatial==2
            edit=take&(before[t]!=after[t]);benefits.append(dict(scope=scope,period=t,changed=int(edit.sum()),fixed=int((edit&(before[t]!=gt[t])&(after[t]==gt[t])).sum()),harmed=int((edit&(before[t]==gt[t])&(after[t]!=gt[t])).sum()),changed_still_wrong=int((edit&(before[t]!=gt[t])&(after[t]!=gt[t])).sum())))
            yy.append(gt[t][take].astype(np.int64)-1)
            pp['previous_V85'].append(before[t][take].astype(np.int64)-1);pp['root_axis_V88'].append(after[t][take].astype(np.int64)-1)
        results[scope]={}
        for name,p in pp.items():
            m=metric(np.concatenate(yy),np.concatenate(p));results[scope][name]=m
            for r in m['classes']:rows.append(dict(scope=scope,reference='unchanged_user_corrected_V87',prediction=name,**r))
    csvout(OUT/'Temporal_class_metrics_same_V87_reference.csv',rows);csvout(OUT/'Reference_agreement_edit_benefits_and_harms.csv',benefits);js(OUT/'evaluation_results.json',results)
    # Keep the old 158 confirmed-root records as a separate audit denominator.
    rr=[]
    for t in PERIODS[1:]:
        px=dict(np.load(DATA/'unet'/(t+'_pixels.npz')));use=px['split']==2
        for rid in np.unique(px['rid'][use]):
            if rid<0:continue
            take=use&(px['rid']==rid)
            if take.sum()<10:continue
            r=dict(period=t,reference_root_target=int(rid),pixels=int(take.sum()))
            for name,a in [('previous_V85',before[t]),('root_axis_V88',after[t])]:
                v=a.ravel()[px['pixel'][take]];r[name+'_new_extension_conflict5']=bool(min((v==2).sum(),(v==3).sum())>=5)
            rr.append(r)
    csvout(OUT/'Historical_known_root_identity_audit.csv',rr)
    cases=[dict(key='A',period='T2',ROI=[1290,220,1540,470],point=[1375,320],explanation='Existing T1 root: false new-colored distal shaft becomes extension.'),dict(key='B',period='T2',ROI=[1530,590,1830,850],point=[1750,686],explanation='First-observed axis: false persistent / extension intervals become new.'),dict(key='C',period='T3',ROI=[1100,205,1290,385],point=[1210,267],explanation='Two roots contact: upper root already visible in T2; horizontal root first appears in T3.'),dict(key='D',period='T3',ROI=[1005,550,1165,715],point=[1085,607],explanation='Existing root: false new interval becomes growth; old shaft remains persistent.')]
    caseaudit=[]
    for c in cases:
        t=c['period'];xy=np.array(c['point']);pts=snapshots[t]['points'];ai=int(np.argmin([np.min(np.linalg.norm(p-xy,axis=1)) for p in pts]));state=snapshots[t]['states'][ai];owner=np.load(OUT/'ownership'/(t+'_axis_owner.npz'))['axis'];use=owner==ai
        caseaudit.append(dict(case=c['key'],period=t,axis=ai,state=state['state'],ROI=c['ROI'],explanation=c['explanation'],before_class_pixels={str(k):int((before[t][use]==k).sum()) for k in [1,2,3]},after_class_pixels={str(k):int((after[t][use]==k).sum()) for k in [1,2,3]},changed_pixels_on_axis=int((use&(before[t]!=after[t])).sum()),history=state))
    js(OUT/'User_case_audit.json',caseaudit)
    structural=dict(confident_axis_records=len(identityrows),confident_new_axes=sum(r['state']=='new' for r in identityrows),confident_established_axes=sum(r['state']=='established' for r in identityrows),forbidden_class_pixels_before=sum(r['forbidden_before'] for r in identityrows),forbidden_class_pixels_after=sum(r['forbidden_after'] for r in identityrows),historical_reference_root_records=len(rr),historical_known_root_new_extension_mixes={name:sum(r[name+'_new_extension_conflict5'] for r in rr) for name in ['previous_V85','root_axis_V88']},all_axis_paths=sum(len(snapshots[t]['points']) for t in PERIODS),unresolved_axis_states=sum(r['state']=='uncertain' for t in PERIODS[1:] for r in snapshots[t]['states']),note='Accepted-axis state checks are a decoder invariant, not independent accuracy. Historical human-root IDs can span different inferred axes; their separate audit is retained.')
    js(OUT/'structural_audit.json',structural)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);manifest=[]
    cv=Canvas(3400,3950,figdir);cv.text(1700,25,'U-Net: actual repairs in the reported examples',56);cv.text(1700,110,'Same root geometry | Full-axis history and connection ownership | Predictions are changed',31)
    xx=[150,960,1770,2580]
    for x,label in zip(xx,['RGB','Previous V85','Root-axis V88','Changed pixels']):cv.text(x+340,210,label,36)
    for i,c in enumerate(cases):
        t=c['period'];x0,y0,x1,y1=c['ROI'];top=285+i*870;cv.text(25,top+320,c['key']+' '+t,40,'lt')
        a=before[t][y0:y1,x0:x1];b=after[t][y0:y1,x0:x1];diff=np.full((*a.shape,3),255,np.uint8);diff[a>0]=[215,215,215];diff[a!=b]=[213,94,0]
        for x,img in zip(xx,[rgb[t][y0:y1,x0:x1],isolated(a),isolated(b),diff]):
            h,w=img.shape[:2];scale=min(680/w,700/h);ww,hh=int(w*scale),int(h*scale);cv.panel(img,x+(680-ww)//2,top,ww,hh,x!=xx[0])
        cv.text(1700,top+730,c['explanation'],28)
    cv.legend(3820);manifest.append(cv.save('Fig1_User_examples_actual_prediction_repairs'))
    cv=Canvas(4100,3820,figdir);cv.text(2050,25,'Four-period U-Net root-axis temporal consistency',60);cv.text(2050,115,'Reference unchanged from V87 | Learned ConvLSTM + GAT features with structured axis decoding',31)
    xxx=[130,1130,2130,3130];w=850;h=371
    for x,s in zip(xxx,['RGB','Current manual reference','Previous V85','Root-axis V88']):cv.text(x+w//2,210,s,33)
    for i,t in enumerate(PERIODS):
        y=280+i*825;cv.text(25,y+320,t,44,'lt');cv.panel(rgb[t],xxx[0],y+180,w,h)
        for x,a in zip(xxx[1:],[gt[t],before[t],after[t]]):cv.panel(overlay(rgb[t],a),x,y,w,h);cv.panel(isolated(a),x,y+h+10,w,h,True)
    cv.legend(3700);manifest.append(cv.save('Fig2_Four_period_root_axis_comparison'))
    # The user-confirmed numbered identities below are audit DISPLAY only and
    # are read AFTER immutable prediction hashes have been frozen above.
    doc=json.loads((REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json').read_text())
    x0,y0,x1,y1=cases[2]['ROI'];cv=Canvas(3260,4300,figdir);cv.text(1630,25,'Example C: root identities explain the contact',54);cv.text(1630,110,'N-0043: observed in T2 | N-0041: first observed in T3 | Identity overlay is annotation audit only',29)
    for i,t in enumerate(PERIODS):
        top=220+i*950;cv.text(25,top+340,t,43,'lt');frame=next(f for f in doc['root_structure_annotation']['periods'] if f['period']==t);r=frame['owner_map_rle'];owner=np.repeat(np.array(r[::2],np.int32),np.array(r[1::2],np.int64)).reshape(1108,2541);aids=[s['index'] for s in frame['segments'] if s['root_id']=='R20473'];bids=[s['index'] for s in frame['segments'] if s['root_id']=='R20882']
        identity=np.full((*owner.shape,3),255,np.uint8);identity[after[t]>0]=[205,205,205];identity[np.isin(owner,aids)&(after[t]>0)]=[0,80,160];identity[np.isin(owner,bids)&(after[t]>0)]=[213,94,0]
        for x,label,img in zip([130,1180,2230],['RGB','Temporal prediction V88','Numbered-root audit'],[rgb[t],isolated(after[t]),identity]):
            cv.text(x+450,top,label,32);cv.panel(img[y0:y1,x0:x1],x,top+60,900,853,x!=130)
    # Each panel has a 180/190 ratio; width900 implies height853.
    cv.text(1630,4150,'Blue identity: N-0043 | Orange identity: N-0041 | Temporal yellow/green do not imply one biological root.',27)
    manifest.append(cv.save('Fig3_Example_C_four_period_identity_audit'))
    # Separate chronology for the three corrected axes prevents reliance on a favorable T3 crop.
    for c in [cases[0],cases[1],cases[3]]:
        cv=Canvas(3260,3850,figdir);cv.text(1630,25,'Example '+c['key']+': same native root window across all periods',52);cv.text(1630,110,'Previous prediction vs actual root-axis repair | Fixed binary segmentation',30);x0,y0,x1,y1=c['ROI']
        for i,t in enumerate(PERIODS):
            top=220+i*850;cv.text(25,top+330,t,42,'lt')
            for x,label,img in zip([130,1180,2230],['RGB','Previous V85','Root-axis V88'],[rgb[t],isolated(before[t]),isolated(after[t])]):
                cv.text(x+450,top,label,32);h,w=img[y0:y1,x0:x1].shape[:2];scale=min(900/w,700/h);ww,hh=int(w*scale),int(h*scale);cv.panel(img[y0:y1,x0:x1],x+(900-ww)//2,top+60,ww,hh,x!=130)
        cv.legend(3720);manifest.append(cv.save('Fig4_Example_'+c['key']+'_four_period_chronology'))
    js(OUT/'figure_manifest.json',dict(files=manifest,dpi=600,black_bold_Times_New_Roman=True,categorical_nearest_neighbor=True,full_field_four_periods=True))
    for p,digest in verify['immutable_inputs_sha256'].items():assert sha(p)==digest
    for t,h in verify['prediction_sha256'].items():assert sha(OUT/'full'/(t+'_classes.png'))==h
    js(OUT/'evaluation_verification.json',dict(passed=True,reference_V87_unchanged=True,old_weights_and_predictions_unchanged=True,new_prediction_classes_really_changed=True,binary_masks_bitwise_identical=True,all_four_full_fields_processed=True,manual_ids_display_only_after_prediction_freeze=True,prediction_sha256=verify['prediction_sha256']))
    text='# Root-axis temporal consistency: prediction repair, not another reference edit\n\n'
    text+='This changes the U-Net temporal prediction maps on all four registered upper fields, starting from V85. It does NOT change the user reference, root masks, widths, lengths, original CNN weights, ConvLSTM/GAT weights, or historical predictions. No new optimization epochs are claimed. Existing learned ConvLSTM/GAT identity and state features are reused with explicit chronology and axis decoding. This is learned-feature inference plus structural decoding, not a claim that neural training alone fixed the examples.\n\n'
    text+='A prior root may have a short old shaft and a much longer later extension. Its new-to-old length fraction cannot create a second birth. Same-prior-root arm continuity can correct junction pairing even when the continuing mother bends more than a new lateral. A first-observed root axis can contain false persistent as well as false extension islands: current state-head evidence, absent earlier aligned roots and later same-axis visibility support a whole-axis new state. An established root can contain persistent shaft and genuine distal extension, with approximate earlier-tip boundaries recorded when uncertain. Shared pixels have one incident axis owner; a true new lateral is separate from its mother. A growing tip merely touching another root is not automatically a daughter.\n\n'
    text+='The old arbitrary .55 same-root acceptance gate is replaced by the inherited validation-calibrated connection threshold. No threshold is selected for a target Dice. Previous root lengths and contours are frozen. The final refinement uses all four fields including later visibility; it is offline, not future-free streaming inference. Future visibility alone never proves earlier presence. Some short/occluded/ambiguous axes remain unresolved. All accepted new/established axes satisfy their semantic state invariant over the owned native-width pixels, including halos. This invariant is NOT independent accuracy; the ownership inference can still be wrong.\n\n'
    text+='Known examples:\n\n'
    for c in caseaudit:text+='- '+c['case']+': '+c['explanation']+' Changed pixels on identified axis: '+str(c['changed_pixels_on_axis'])+'.\n'
    text+='\nExample C is explicitly retained as two touching roots: the diagonal root is already visible in T2 and is manually numbered N-0043; the horizontal root first appears in T3 and is N-0041. Root-ID overlays are audit display ONLY, loaded after predictions were frozen. No inference reads these root IDs. Their temporal colors can legitimately meet at a contact; they must not be merged to manufacture uniformity. Example D separates the older0002 axis from the N-0042 axis first observed in T2; false new coloring on the older axis is repaired, while genuine distal growth is preserved.\n\n'
    text+='The V87 reference remains byte-for-byte unchanged: earlier two T2 reference corrections and all T4 extension-to-new changes are preserved. Metrics are reference agreement under this SAME reference. All B3 is a development position; these user-reported examples include previously audited spatial areas, so old inner-test scores are now described as spatial audit, NOT independent validation. Scores are conditional on fixed-root foreground and exclude binary segmentation misses and T1 baseline.\n\n'
    text+='| Scope | Prediction | Persistent | New | Extension | Macro |\n|---|---|---:|---:|---:|---:|\n'
    for scope,models in results.items():
        for name,m in models.items():text+='| '+scope+' | '+name+' | '+' | '.join(f"{r['Dice']:.6f}" for r in m['classes'])+f" | {m['macro_Dice']:.6f} |\n"
    text+='\nRoot state and historical confirmed-root checks (different denominators):\n\n```json\n'+json.dumps(structural,indent=2)+'\n```\n'
    (OUT/'RESULTS.md').write_text(text)
    (OUT/'source').mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    js(OUT/'completion.json',dict(state='complete',actual_prediction_repairs=True,no_training=True,reference_unchanged=True,all_four_periods=True,unresolved_cases_retained=True))
    print(json.dumps(dict(changes=pixrows,structural=structural,metrics=results,cases=caseaudit),indent=2),flush=True)


if __name__=='__main__':main()
