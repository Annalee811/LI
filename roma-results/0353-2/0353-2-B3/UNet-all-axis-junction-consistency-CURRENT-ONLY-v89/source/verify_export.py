"""Check scalar segment/junction colors; export CURRENT-ONLY figures and paired scores."""
import csv,json,pickle,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_all_axes_and_crossing_consistency_v89_20261029'
PREV=ROOT/'results/B3_root_state_prediction_repair_v88_20261028'
V85=ROOT/'results/B3_branch_continuity_review_v85_20261028'
REF=ROOT/'results/B3_T4_all_extension_reference_to_new_v87_20261028'
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
DATA=ROOT/'results/B3_original_UNet_latestCG_upload24_v82_20261028/data'
PERIODS=['T1','T2','T3','T4']
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from topology import fuse_crossing_bridges
from prepare import js,csvout,sha
from evaluate import metric
from rendering import Canvas,isolated,overlay


def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    verify=json.loads((OUT/'consistency_verification.json').read_text());assert verify['passed']
    after={t:np.array(Image.open(OUT/'full'/(t+'_classes.png'))) for t in PERIODS};before={t:np.array(Image.open(PREV/'full'/(t+'_classes.png'))) for t in PERIODS}
    with (ROOT/'results/B3_root_axis_junction_v83_20261028/data/topology.pkl').open('rb') as f:top=pickle.load(f)
    ev=read(OUT/'Every_junction_event_audit.csv');segment_rows=[];event_rows=[]
    for t in PERIODS:
        assert np.array_equal(after[t]>0,before[t]>0)
        if t=='T1':assert np.array_equal(after[t],before[t]);assert np.isin(after[t],[0,4]).all();continue
        tp=top['topos'][t];fuse_crossing_bridges(tp);core=np.zeros(after[t].shape,bool);footprint=np.zeros(after[t].shape,bool);native_half_width=ndi.distance_transform_edt(after[t]>0)
        for h in tp['hubs']:
            if h.get('disabled') or len(h['arms'])<3:continue
            p=h['points'];core[p[:,1],p[:,0]]=True;radius=int(np.clip(np.ceil(4*np.median(native_half_width[p[:,1],p[:,0]])),4,10));x0=max(0,int(p[:,0].min())-radius);x1=min(after[t].shape[1],int(p[:,0].max())+radius+1);y0=max(0,int(p[:,1].min())-radius);y1=min(after[t].shape[0],int(p[:,1].max())+radius+1);local=np.ones((y1-y0,x1-x0),bool);local[p[:,1]-y0,p[:,0]-x0]=False;footprint[y0:y1,x0:x1]|=(ndi.distance_transform_edt(local)<=radius)&(after[t][y0:y1,x0:x1]>0)
        assert np.array_equal(footprint,np.array(Image.open(OUT/'changes'/(t+'_junction_footprint.png')))>0);groups,_=ndi.label(footprint,np.ones((3,3)))
        for i,s in enumerate(tp['segments']):
            if s.get('absorbed'):continue
            p=s['points'];p=p[~footprint[p[:,1],p[:,0]]];v=after[t][p[:,1],p[:,0]]
            bad_new_growth=bool((v==2).any() and(v==3).any());bad_new_old=bool((v==2).any() and(v==1).any())
            segment_rows.append(dict(period=t,segment=i,body_pixels=len(v),new_extension=bad_new_growth,new_persistent=bad_new_old));assert not bad_new_growth and not bad_new_old
        for r in [r for r in ev if r['period']==t]:
            use=groups==int(r['event']);allowed=[int(k) for k in r['allowed_types'].split(',')];bad=int((use&~np.isin(after[t],allowed)).sum());assert bad==0
            event_rows.append(dict(period=t,event=int(r['event']),checked_pixels=int(use.sum()),allowed_types=r['allowed_types'],forbidden_after=bad))
    csvout(OUT/'Independent_scalar_segment_check.csv',segment_rows);csvout(OUT/'Independent_junction_pixel_check.csv',event_rows)
    js(OUT/'independent_consistency_check.json',dict(passed=True,unbranched_segments_checked=len(segment_rows),new_extension_segments=0,new_persistent_segments=0,junction_events_checked=len(event_rows),forbidden_junction_pixels=0,T1_baseline_unchanged=True,foreground_bitwise_unchanged=True,scope='Independent scalar-color implementation on the same inferred topology; not independent biological ground truth.'))
    js(OUT/'prediction_freeze.json',dict(prediction_sha256=verify['prediction_sha256'],freeze_before_reference_scores=True,reference_sha256=sha(REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json')))
    # Reference data enter only after prediction and consistency verification.
    gt={t:np.array(Image.open(REF/'corrected_reference'/(t+'_classes.png'))) for t in PERIODS};spatial=np.load(DATA/'spatial.npy');results={};rows=[];benefits=[]
    for scope in ['original_inner_spatial_audit','all_valid_annotated_root_pixels']:
        yy=[];pp={name:[] for name in ['previous_V88','current_V89']}
        for t in PERIODS[1:]:
            use=np.isin(gt[t],[1,2,3])&(before[t]>0)
            if scope=='original_inner_spatial_audit':use&=spatial==2
            changed=use&(before[t]!=after[t]);benefits.append(dict(scope=scope,period=t,changed=int(changed.sum()),fixed=int((changed&(before[t]!=gt[t])&(after[t]==gt[t])).sum()),harmed=int((changed&(before[t]==gt[t])&(after[t]!=gt[t])).sum()),changed_still_wrong=int((changed&(before[t]!=gt[t])&(after[t]!=gt[t])).sum())))
            yy.append(gt[t][use].astype(np.int64)-1);pp['previous_V88'].append(before[t][use].astype(np.int64)-1);pp['current_V89'].append(after[t][use].astype(np.int64)-1)
        results[scope]={}
        for name,p in pp.items():
            m=metric(np.concatenate(yy),np.concatenate(p));results[scope][name]=m
            for r in m['classes']:rows.append(dict(scope=scope,reference='unchanged_V87',prediction=name,**r))
    csvout(OUT/'Temporal_Dice_same_reference_V87.csv',rows);csvout(OUT/'Reference_agreement_benefits_and_harms.csv',benefits);js(OUT/'evaluation_results.json',results)
    old85={t:np.array(Image.open(V85/'full'/(t+'_classes.png'))) for t in PERIODS}
    with (PREV/'root_axis_snapshots.pkl').open('rb') as f:snap=pickle.load(f)
    requested=[dict(name='1_long_axis',t='T2',point=[920,908],ROI=[620,740,1068,1000]),dict(name='2_old_root_distal_axis',t='T2',point=[1375,320],ROI=[1290,150,1540,470]),dict(name='3_new_root_tip',t='T2',point=[1750,686],ROI=[1530,590,1830,850])]
    version=[]
    for c in requested:
        t=c['t'];p=snap[t]['points'];ai=int(np.argmin([np.min(np.linalg.norm(q-c['point'],axis=1)) for q in p]));owner=np.load(OUT/'ownership'/(t+'_axis_owner.npz'))['axis'];use=owner==ai
        version.append(dict(case=c['name'],axis=ai,period=t,ROI=c['ROI'],V85_pixel_counts={str(k):int((old85[t][use]==k).sum()) for k in [1,2,3]},V88_pixel_counts={str(k):int((before[t][use]==k).sum()) for k in [1,2,3]},V89_pixel_counts={str(k):int((after[t][use]==k).sum()) for k in [1,2,3]},earlier_root_evidence=snap[t]['states'][ai]))
    js(OUT/'Requested_coordinates_version_audit.json',version)
    figdir=OUT/'figures';figdir.mkdir(exist_ok=True);manifest=[];rgb={t:np.array(Image.open(BASE/'data/upper'/(t+'_RGB.png'))) for t in PERIODS}
    cv=Canvas(4800,3900,figdir);cv.text(2400,25,'CURRENT PREDICTIONS ONLY - V89',90);cv.text(2400,155,'U-Net temporal output | No old comparison column and no manual-reference column',42)
    xx=[170,1720,3270];ww=1360;hh=593
    for x,label in zip(xx,['Original RGB','CURRENT temporal overlay','CURRENT root classes']):cv.text(x+ww//2,270,label,44)
    for i,t in enumerate(PERIODS):
        y=355+i*790;cv.text(35,y+260,t,58,'lt')
        for x,a in zip(xx,[rgb[t],overlay(rgb[t],after[t]),isolated(after[t])]):cv.panel(a,x,y,ww,hh,x==xx[2])
    cv.legend(3710);manifest.append(cv.save('Fig1_CURRENT_ONLY_four_period_predictions_V89'))
    cv=Canvas(4200,1900,figdir);cv.text(2100,25,'CURRENT LOCAL PREDICTIONS ONLY - V89',75);cv.text(2100,150,'Same three requested locations | Main axes are traced beyond the original screenshot crop',36)
    for i,c in enumerate(requested):
        x0,y0,x1,y1=c['ROI'];x=120+i*1380;cv.text(x+600,300,'Location '+str(i+1)+' / '+c['t'],44);img=isolated(after[c['t']][y0:y1,x0:x1]);h,w=img.shape[:2];scale=min(1200/w,1000/h);w1,h1=int(w*scale),int(h*scale);cv.panel(img,x+(1200-w1)//2,420,w1,h1,True)
    cv.text(2100,1580,'Locations1-2: an earlier root keeps persistent shaft / distal extension. Location3: the new axis is new only.',33)
    cv.legend(1760);manifest.append(cv.save('Fig2_CURRENT_ONLY_requested_locations_V89'))
    # A separately named diagnostic makes OLD and CURRENT unmistakable.
    cv=Canvas(3700,3200,figdir);cv.text(1850,25,'DIAGNOSTIC VERSION CHECK - NOT THE MAIN RESULT',58)
    xxx=[160,1390,2620]
    for x,label in zip(xxx,['V85 OLD comparison column','V88 previous corrected output','V89 CURRENT output']):cv.text(x+500,160,label,36)
    compact=[[768,800,1068,1100],[1290,220,1540,470],[1530,590,1830,850]]
    for i,roi in enumerate(compact):
        x0,y0,x1,y1=roi;top=260+i*900
        for x,a in zip(xxx,[old85['T2'],before['T2'],after['T2']]):
            img=isolated(a[y0:y1,x0:x1]);h,w=img.shape[:2];scale=min(1000/w,750/h);ww1,hh1=int(w*scale),int(h*scale);cv.panel(img,x+(1000-ww1)//2,top,ww1,hh1,True)
        cv.text(1850,top+790,'Requested location '+str(i+1),33)
    cv.legend(3040);manifest.append(cv.save('Diagnostic_V85_V88_V89_requested_coordinates'))
    # Show EVERY previously forbidden new/persistent event, not selected by Dice.
    candidates=[r for r in ev if 'new/persistent' in r['rule'] and int(r['forbidden_before'])>0]
    for page in range((len(candidates)+7)//8):
        cv=Canvas(3100,3840,figdir);cv.text(1550,25,'ALL previously forbidden new/persistent junctions - page '+str(page+1),46);cv.text(1550,110,'OLD V88 on left; CURRENT V89 on right in each pair | All events, not favorable-score selection',29)
        for i,r in enumerate(candidates[page*8:(page+1)*8]):
            t=r['period'];x,y=int(r['x']),int(r['y']);x0=max(0,x-38);y0=max(0,y-38);x1=min(2541,x+39);y1=min(1108,y+39);left=90+(i%2)*1500;top=220+(i//2)*870;cv.text(left+680,top,t+' event '+r['event']+' / '+str((x,y)),32)
            for xc,img,label in [(left,before[t],'OLD V88'),(left+730,after[t],'CURRENT V89')]:
                cv.text(xc+320,top+60,label,28);cv.panel(isolated(img[y0:y1,x0:x1]),xc,top+115,640,640,True)
        cv.legend(3710);manifest.append(cv.save('All_new_persistent_junction_checks_page_'+str(page+1)))
    js(OUT/'figure_manifest.json',dict(files=manifest,dpi=600,black_bold_Times_New_Roman=True,main_figures_current_predictions_only=True,old_versions_confined_to_explicit_diagnostic_figures=True))
    for p,digest in verify['immutable_input_sha256'].items():assert sha(p)==digest
    for t,digest in verify['prediction_sha256'].items():assert sha(OUT/'full'/(t+'_classes.png'))==digest
    text='# Every inferred mixed axis and junction event: V89\n\n'
    text+='This is an inference/structured-decoding correction on the existing U-Net + ConvLSTM/GAT result, not additional training. V88 omitted206 low/unknown-state mixed axes from its confident-state invariant, including58 axes containing both new and extension. All206 are now explicitly processed rather than omitted. The learned state head, current predicted shaft evidence and multi-period correspondence determine a MAP new/established state.59 mixed axes retain low decision-margin confidence and are exported separately; these scores are not calibrated accuracy probabilities. Root identity can still be wrong.\n\n'
    text+='Every inferred new axis is new only. Every inferred older axis permits persistent shaft and/or extension but no second new birth. Current foreground, native root widths/lengths and T1 baseline remain bitwise unchanged. The previous V87 manual reference and every historical model/result file are retained unchanged. Manual classes and root IDs are not prediction inputs. No target Dice or synthetic error is used.\n\n'
    text+='All shared-core footprints are grouped into junction events. The neighborhood spans two local native root diameters (radius4..10px), so outside-edge halos are checked too. New + persistent incident roots prohibit extension in the shared footprint; new + extension prohibit a persistent third type. Same-type crossings retain that type. If a forbidden extension pixel was owned by the growing neighbor but belongs to a persistent crossing strand, ownership is reassigned to the appropriate incident persistent root, rather than painting a purple island into the growing root. Tiny disconnected pieces can join a uniquely nearest new axis only with learned new-state evidence, gap<=8px and a distance margin over older axes; foreground gaps are not filled. This changes local instance ownership but never the combined foreground mask. Complex overlapping events follow the explicitly requested shared-footprint rule; biological identities at ambiguous contacts remain uncertain.\n\n'
    text+='A separate scalar-color implementation verifies every non-junction unbranched segment and every junction footprint after decoding. These are consistency checks on inferred topology, not independent biological accuracy. The original spatial audit region is a known development region. Conditional temporal Dice excludes binary root misses and T1 baseline, and the same unchanged V87 reference is used before and after.\n\n'
    text+='The three requested coordinate examples show mixed old V85 classes but already different V88 data. The earlier four-period comparison included OLD V85 and manual-reference columns. To remove this display ambiguity, the MAIN figures now contain ONLY current V89 predictions. Old versions appear exclusively in a figure explicitly titled DIAGNOSTIC VERSION CHECK.\n\n'
    text+='Consistency summary:\n\n```json\n'+json.dumps({k:v for k,v in verify.items() if k not in ['immutable_input_sha256','prediction_sha256']},indent=2)+'\n```\n\n'
    text+='| Scope | Variant | Persistent | New | Extension | Macro |\n|---|---|---:|---:|---:|---:|\n'
    for scope,models in results.items():
        for name,m in models.items():text+='| '+scope+' | '+name+' | '+' | '.join(f"{r['Dice']:.6f}" for r in m['classes'])+f" | {m['macro_Dice']:.6f} |\n"
    (OUT/'RESULTS.md').write_text(text)
    (OUT/'source').mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    js(OUT/'completion.json',dict(state='complete',current_only_main_figures=True,no_training=True,all_mixed_inferred_axes_processed=True,unchanged_reference_and_geometry=True,low_confidence_cases_retained_for_review=True))
    print(json.dumps(dict(consistency={k:v for k,v in verify.items() if k not in ['immutable_input_sha256','prediction_sha256']},independent_check=json.loads((OUT/'independent_consistency_check.json').read_text()),metrics=results,main_figures=manifest[:2],junction_gallery_events=len(candidates)),indent=2),flush=True)


if __name__=='__main__':main()
