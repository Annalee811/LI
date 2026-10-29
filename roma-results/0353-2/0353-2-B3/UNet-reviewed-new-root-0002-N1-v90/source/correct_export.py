"""Explicitly reviewed correction of ONE T2 new lateral, not a page-wide repaint."""
import json,pickle,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_user_confirmed_new_axis_correction_20261029'
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
BEFORE=ROOT/'results/B3_all_axes_and_crossing_consistency_v89_20261029'
REF=ROOT/'results/B3_T4_all_extension_reference_to_new_v87_20261028'
PERIODS=['T1','T2','T3','T4']
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from prepare import js,sha,csvout
from evaluate import metric
from rendering import Canvas,isolated,overlay


def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True)
    for name in ['full','ownership','figures','source']: (OUT/name).mkdir(exist_ok=True)
    source=REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json'
    files=[source,ROOT/'results/B3_root_axis_junction_v83_20261028/joint_best.pt',BASE/'unet/best_temporal_head.pt']+[p/(t+'_classes.png') for p in [BEFORE/'full',BASE/'unet/full'] for t in PERIODS]
    immutable={str(p):sha(p) for p in files}
    old={t:np.array(Image.open(BEFORE/'full'/(t+'_classes.png'))) for t in PERIODS};new={t:a.copy() for t,a in old.items()}
    owners={t:np.load(BEFORE/'ownership'/(t+'_axis_owner.npz'))['axis'].copy() for t in PERIODS}
    # The user just confirmed this specific root, and their existing instance
    # annotation independently identifies the entire curve as 0002-N1 / R20484.
    # This is REVIEW SUPERVISION, deliberately not a label-blind model inference.
    doc=json.loads(source.read_text());st=doc['root_structure_annotation'];track=next(r for r in st['root_track_hierarchy'] if r['root_id']=='R20484')
    assert track['label']=='0002-N1' and track['first_seen_period']=='T2' and track['annotation_status_v60']=='user_defined'
    frame=next(f for f in st['periods'] if f['period']=='T2');r=frame['owner_map_rle'];manual_owner=np.repeat(np.array(r[::2],np.int32),np.array(r[1::2],np.int64)).reshape(1108,2541)
    segment_indices=[s['index'] for s in frame['segments'] if s['root_id']=='R20484']
    target=np.isin(manual_owner,segment_indices)&(old['T2']>0)
    assert int(target.sum())==3149
    yy,xx=np.where(target);bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1];assert bbox==[1271,201,2068,457]
    with (ROOT/'results/B3_root_state_prediction_repair_v88_20261028/root_axis_snapshots.pkl').open('rb') as f:snap=pickle.load(f)
    axis=snap['T2']['points'][162];values=manual_owner[axis[:,1],axis[:,0]];on=np.isin(values,segment_indices);cut=int(np.flatnonzero(on)[0]);assert cut==23 and on[cut:].all()
    original_owned=owners['T2'].copy();new_axis=len(snap['T2']['points']);assert new_axis==772
    before_counts={str(k):int((old['T2'][target]==k).sum()) for k in [1,2,3]}
    new['T2'][target]=2;owners['T2'][target]=new_axis
    # Preserve the actual older contact strand, rather than turning it green.
    contact_old=(original_owned==162)&~target
    assert np.array_equal(new['T2'][contact_old],old['T2'][contact_old])
    changed=new['T2']!=old['T2'];assert int(changed.sum())==2100
    assert not changed[~target].any();assert (new['T2'][target]==2).all()
    for t in PERIODS:
        assert np.array_equal(new[t]>0,old[t]>0)
        if t!='T2':assert np.array_equal(new[t],old[t])
        Image.fromarray(new[t]).save(OUT/'full'/(t+'_classes.png'));np.savez_compressed(OUT/'ownership'/(t+'_axis_owner.npz'),axis=owners[t])
    Image.fromarray(target.astype(np.uint8)*255).save(OUT/'Reviewed_root_0002-N1_support.png');Image.fromarray(changed.astype(np.uint8)*255).save(OUT/'T2_changed_pixels.png')
    ownership_rows=[]
    for ai in np.unique(original_owned[target]):
        use=target&(original_owned==ai);ownership_rows.append(dict(period='T2',old_model_axis=int(ai),reviewed_new_axis=new_axis,reviewed_root_label='0002-N1',reassigned_pixels=int(use.sum()),old_axis_pixels_outside_reviewed_root_preserved=int(((original_owned==ai)&~target).sum())))
    csvout(OUT/'Reviewed_root_instance_reassignment.csv',ownership_rows)
    feedback=dict(type='explicit_user_reviewed_prediction_correction',period='T2',manual_root_id='R20484',manual_root_label='0002-N1',class_name='newly_observed',prediction_class_code=2,source_annotation_sha256=sha(source),fixed_root_support_sha256=sha(OUT/'Reviewed_root_0002-N1_support.png'),new_model_axis=new_axis,old_inherited_axis=162,virtual_contact_split_coordinate=axis[cut].tolist(),old_contact_strand_coordinate_range=[axis[0].tolist(),axis[cut-1].tolist()],old_contact_root_id='R20038',biological_parent_root_id=track['parent_root_id'],decision='Same-root continuity among the reviewed new-lateral fragments; biological parent/contact roots are different identities and must not transfer their old state.',must_not_link_as_same_identity=['R20038',track['parent_root_id']],manual_annotation_modified=False,model_retrained=False)
    js(OUT/'Reviewed_connection_and_birth_supervision.json',feedback)
    audit=dict(period='T2',root_id='R20484',root_label='0002-N1',root_first_seen_period='T2',selected_root_foreground_pixels=3149,changed_pixels=2100,bbox=bbox,before_class_counts=before_counts,after_class_counts={'1':0,'2':3149,'3':0},other_root_pixels_changed=0,T1_T3_T4_prediction_pixels_changed=0,older_contact_strand_preserved=True,old_model_axis_count_representing_this_one_root=len(ownership_rows),new_model_axis=new_axis,error_mechanism='Only21 skeleton samples near an old contact strand were accepted as earlier-root evidence; the new lateral body had472 new-class samples. The decoder joined contact and lateral and inherited the old state. The user and existing confirmed root identity specify a new T2 lateral.',correction_scope='ONLY the explicitly confirmed numbered root intersected with the immutable prediction foreground, not a rectangular/page repaint',assisted_review=True,no_training=True)
    js(OUT/'correction_audit.json',audit)
    # Metrics are explicitly annotation-assisted agreement after this correction,
    # not an independent validation of a newly learned model.
    gt={t:np.array(Image.open(REF/'corrected_reference'/(t+'_classes.png'))) for t in PERIODS};spatial=np.load(ROOT/'results/B3_original_UNet_latestCG_upload24_v82_20261028/data/spatial.npy');results={};rows=[]
    for scope in ['original_inner_spatial_audit','all_valid_annotated_root_pixels']:
        yy=[];pp={name:[] for name in ['V89_before_review','V90_after_explicit_user_review']}
        for t in PERIODS[1:]:
            use=np.isin(gt[t],[1,2,3])&(old[t]>0)
            if scope=='original_inner_spatial_audit':use&=spatial==2
            yy.append(gt[t][use].astype(np.int64)-1);pp['V89_before_review'].append(old[t][use].astype(np.int64)-1);pp['V90_after_explicit_user_review'].append(new[t][use].astype(np.int64)-1)
        results[scope]={}
        for name,p in pp.items():
            m=metric(np.concatenate(yy),np.concatenate(p));results[scope][name]=m
            for r in m['classes']:rows.append(dict(scope=scope,reference='unchanged_V87',prediction=name,evaluation_status='annotation_assisted_review_agreement',**r))
    js(OUT/'evaluation_results.json',results);csvout(OUT/'Dice_after_review_same_reference_V87.csv',rows)
    figdir=OUT/'figures';manifest=[]
    def comparison(roi,stem,title):
        x0,y0,x1,y1=roi;cv=Canvas(3600,1500,figdir);cv.text(1800,25,title,60);cv.text(1800,135,'T2 root 0002-N1: explicitly confirmed NEW | Only this root changes; all other roots are preserved',33)
        for xc,label,a in [(150,'V89 BEFORE: incorrect inherited state',old['T2']),(1900,'V90 CURRENT: reviewed NEW root',new['T2'])]:
            cv.text(xc+775,270,label,39);img=isolated(a[y0:y1,x0:x1]);hh,ww=img.shape[:2];scale=min(1550/ww,800/hh);pw,ph=int(ww*scale),int(hh*scale);cv.panel(img,xc+(1550-pw)//2,350,pw,ph,True)
        cv.text(1800,1200,'No root width, length, foreground pixel or manual label is changed. The older contact root stays unchanged.',31);cv.legend(1370);manifest.append(cv.save(stem))
    comparison([1550,180,2090,440],'Fig1_Reported_new_root_before_after','User-confirmed new root: correction of mistaken old-root inheritance')
    comparison([1220,150,2120,520],'Fig2_Entire_numbered_new_root_before_after','Entire numbered root 0002-N1: continuous new-root identity')
    rgb={t:np.array(Image.open(BASE/'data/upper'/(t+'_RGB.png'))) for t in PERIODS}
    cv=Canvas(4800,3900,figdir);cv.text(2400,25,'CURRENT REVIEWED PREDICTIONS ONLY - V90',80);cv.text(2400,155,'Only the user-confirmed T2 root 0002-N1 is corrected; T1, T3 and T4 remain unchanged',38);xx=[170,1720,3270];w=1360;h=593
    for x,label in zip(xx,['Original RGB','CURRENT temporal overlay','CURRENT root classes']):cv.text(x+w//2,270,label,44)
    for i,t in enumerate(PERIODS):
        top=355+i*790;cv.text(35,top+260,t,58,'lt')
        for x,a in zip(xx,[rgb[t],overlay(rgb[t],new[t]),isolated(new[t])]):cv.panel(a,x,top,w,h,x==xx[2])
    cv.legend(3710);manifest.append(cv.save('Fig3_CURRENT_ONLY_four_period_predictions_V90'))
    js(OUT/'figure_manifest.json',dict(files=manifest,dpi=600,black_bold_Times_New_Roman=True,categorical_nearest_neighbor=True,original_user_report_ROI=[1550,180,2090,440],whole_reviewed_root_ROI=[1220,150,2120,520]))
    for p,digest in immutable.items():assert sha(p)==digest
    js(OUT/'verification.json',dict(passed=True,only_one_user_confirmed_root_modified=True,other_root_pixels_changed=0,T1_T3_T4_unchanged=True,older_contact_root_unchanged=True,all_reviewed_root_pixels_are_new=True,whole_foreground_bitwise_identical=True,manual_annotation_unchanged=True,no_training=True,not_label_blind_inference=True,immutable_input_sha256=immutable,prediction_sha256={t:sha(OUT/'full'/(t+'_classes.png')) for t in PERIODS}))
    text='# Explicit user-reviewed correction of one new T2 root\n\n'
    text+='The user confirmed the long curved root shown as extension is entirely new in T2, and clarified that this instruction applies to THIS root, not all roots on the page. Its existing confirmed annotation identity is0002-N1/R20484, a new lateral first seen in T2. V89 represented that one root across11 model ownership IDs and wrongly inherited old state through a contact with R20038. In axis162, the original U-Net has26 persistent centerline pixels followed by489 new pixels. Only the first21 centerline points match the older T1 strand; that small contact prefix must not define the state of the lateral body. The split is at (2067,203), and the true older contact strand is retained.\n\n'
    text+='This is an EXPLICITLY ANNOTATION-ASSISTED USER REVIEW CORRECTION, not additional neural training or a new label-blind model inference. The existing confirmed numbered-root ownership provides the scope, intersected with the immutable predicted foreground. Exactly3149 selected-root pixels become new;1049 were already new,1834 extension and266 persistent pixels are corrected, for2100 actual class changes. No rectangular region or entire page is repainted. Model ownership of the selected root fragments is unified as reviewed axis772. Other root pixels, manual labels, model weights, all binary root silhouettes and all T1/T3/T4 predictions remain bitwise unchanged. No gaps are filled and no root pixels are added or removed.\n\n'
    text+='The reviewed connection/identity supervision is saved separately for future model training: fragments of this numbered lateral are same-root; its biological parent and the old strand it contacts are different root identities and must not transfer their established-state label. Original files and all V89 results are preserved. Future inference should retain this explicitly reviewed example rather than rerunning the earlier false-contact inheritance rule.\n\n'
    text+='Same-reference scores are descriptive agreement after an annotation-assisted correction, NOT independent accuracy or evidence that unchanged network weights learned a new capability. Both original inner spatial audit and all valid annotated foreground scopes are retained; T1 and binary root misses are excluded from three-class temporal scores. Reference V87, including earlier T2 and T4 corrections, is unchanged.\n\n'
    text+='| Scope | Variant | Persistent | New | Extension | Macro |\n|---|---|---:|---:|---:|---:|\n'
    for scope,models in results.items():
        for name,m in models.items():text+='| '+scope+' | '+name+' | '+' | '.join(f"{r['Dice']:.6f}" for r in m['classes'])+f" | {m['macro_Dice']:.6f} |\n"
    text+='\nDownloads:\n\n'+''.join('- ['+r['file']+'](figures/'+r['file']+')\n' for r in manifest)
    (OUT/'RESULTS.md').write_text(text)
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    js(OUT/'completion.json',dict(state='complete',explicit_user_review_only=True,target='T2 0002-N1',changed_pixels=2100,no_training=True,unrelated_roots_and_geometry_unchanged=True));print(json.dumps(dict(audit=audit,metrics=results),indent=2),flush=True)


if __name__=='__main__':main()
