"""Direct original U-Net V66 vs V89 comparison; no intermediate baseline substitution."""
import json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_original_UNet_vs_V89_comparison_20261029'
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
CURRENT=ROOT/'results/B3_all_axes_and_crossing_consistency_v89_20261029'
REF=ROOT/'results/B3_T4_all_extension_reference_to_new_v87_20261028'
PERIODS=['T1','T2','T3','T4']
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from prepare import js,sha,csvout
from rendering import Canvas,isolated,overlay


def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True);figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
    ck=BASE/'unet/best_temporal_head.pt';assert sha(ck)=='58a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382'
    assert json.loads((CURRENT/'completion.json').read_text())['state']=='complete'
    frozen=[ck,REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json']+[p/(t+'_classes.png') for p in [BASE/'unet/full',CURRENT/'full'] for t in PERIODS]+[BASE/'data/upper'/(t+'_RGB.png') for t in PERIODS]
    hashes={str(p):sha(p) for p in frozen}
    rgb={t:np.array(Image.open(BASE/'data/upper'/(t+'_RGB.png'))) for t in PERIODS}
    old={t:np.array(Image.open(BASE/'unet/full'/(t+'_classes.png'))) for t in PERIODS}
    new={t:np.array(Image.open(CURRENT/'full'/(t+'_classes.png'))) for t in PERIODS}
    for t in PERIODS:assert np.array_equal(old[t]>0,new[t]>0)
    assert np.array_equal(old['T1'],new['T1'])
    manifest=[]
    cv=Canvas(4350,4600,figdir);cv.text(2175,25,'Original U-Net vs V89: four-period comparison',72);cv.text(2175,135,'Original U-Net (V66, selected epoch37) | V89 temporal correction | Identical root geometry',35)
    xx=[180,1630,3080];w=1100;h=480
    for x,label in zip(xx,['Original RGB','ORIGINAL U-Net (V66)','OPTIMIZED output (V89)']):cv.text(x+w//2,235,label,43)
    for i,t in enumerate(PERIODS):
        y=325+i*1040;cv.text(35,y+440,t,54,'lt');cv.panel(rgb[t],xx[0],y+250,w,h)
        for x,a in [(xx[1],old[t]),(xx[2],new[t])]:
            cv.panel(overlay(rgb[t],a),x,y,w,h);cv.panel(isolated(a),x,y+h+20,w,h,True)
    cv.legend(4490);manifest.append(cv.save('Fig1_Original_UNet_vs_V89_four_period_comparison'))
    roi=[620,740,1068,1100];x0,y0,x1,y1=roi
    cv=Canvas(4000,3650,figdir);cv.text(2000,25,'Original U-Net vs V89: matched detail across four periods',61);cv.text(2000,125,'Same native ROI in T1-T4: [620, 740, 1068, 1100] | Persistent, new and extension classes',32)
    xxx=[160,1480,2800];maxw=1040;maxh=700
    for i,t in enumerate(PERIODS):
        top=260+i*790;cv.text(25,top+330,t,46,'lt')
        for x,label,a in zip(xxx,['Original RGB','ORIGINAL U-Net (V66)','OPTIMIZED output (V89)'],[rgb[t],isolated(old[t]),isolated(new[t])]):
            cv.text(x+maxw//2,top,label,35);patch=a[y0:y1,x0:x1];hh,ww=patch.shape[:2];s=min(maxw/ww,maxh/hh);pw,ph=int(ww*s),int(hh*s);cv.panel(patch,x+(maxw-pw)//2,top+60,pw,ph,x!=xxx[0])
    cv.legend(3520);manifest.append(cv.save('Fig2_Original_UNet_vs_V89_matched_four_period_detail'))
    rois=[[620,740,1068,1000],[1290,150,1540,470],[1530,590,1830,850]]
    cv=Canvas(4000,3400,figdir);cv.text(2000,25,'Original U-Net vs V89: three reported T2 root regions',63);cv.text(2000,135,'Exact original U-Net baseline | Same native crop and segmentation mask in each comparison',33)
    for i,roi0 in enumerate(rois):
        x0,y0,x1,y1=roi0;top=260+i*980;cv.text(35,top+320,chr(65+i),46,'lt')
        for x,label,a in zip(xxx,['Original RGB','ORIGINAL U-Net (V66)','OPTIMIZED output (V89)'],[rgb['T2'],isolated(old['T2']),isolated(new['T2'])]):
            cv.text(x+520,top,label,35);patch=a[y0:y1,x0:x1];hh,ww=patch.shape[:2];s=min(1040/ww,720/hh);pw,ph=int(ww*s),int(hh*s);cv.panel(patch,x+(1040-pw)//2,top+60,pw,ph,x!=xxx[0])
        cv.text(2000,top+815,'T2 native ROI: '+str(roi0),30)
    cv.legend(3270);manifest.append(cv.save('Fig3_Original_UNet_vs_V89_reported_T2_details'))
    # Extract previously verified exact same-reference metrics; no new model run.
    raw_orig=json.loads((REF/'evaluation_results.json').read_text());raw_new=json.loads((CURRENT/'evaluation_results.json').read_text());metrics={};rows=[]
    for scope,origscope in [('original_inner_spatial_audit','internal_fixed_root_spatial_holdout'),('all_valid_annotated_root_pixels','all_annotated_fixed_root_pixels')]:
        a=raw_orig[origscope]['two_T2_and_all_T4_corrections_V87']['T2_T3_T4_combined']['original_U_Net_V66'];b=raw_new[scope]['current_V89']
        assert [r['support'] for r in a['classes']]==[r['support'] for r in b['classes']]
        assert sum(map(sum,a['confusion']))==sum(map(sum,b['confusion']))
        metrics[scope]={'original_U_Net_V66':a,'optimized_V89':b}
        for name,m in metrics[scope].items():
            for row in m['classes']:rows.append(dict(scope=scope,reference='unchanged_V87',prediction=name,**{k:row[k] for k in ['name','Precision','Recall','Dice','IoU','support']}))
    csvout(OUT/'Original_UNet_vs_V89_same_reference_metrics.csv',rows);js(OUT/'metrics.json',metrics)
    js(OUT/'figure_manifest.json',dict(files=manifest,dpi=600,black_bold_Times_New_Roman=True,categorical_nearest_neighbor=True,original_baseline='V66 original U-Net, selected epoch37; historical reference20 new-root Dice0.8229',comparison='Original U-Net vs actual V89; no V85/V88 or manual reference column',matched_four_period_ROI=[620,740,1068,1100],T2_detail_ROIs=rois))
    for p,digest in hashes.items():assert sha(p)==digest
    js(OUT/'verification.json',dict(passed=True,no_training=True,no_prediction_or_reference_edit=True,original_baseline_checkpoint_verified=True,root_foreground_bitwise_identical=True,T1_identical=True,same_metric_reference_V87=True,same_metric_supports=True,immutable_input_sha256=hashes))
    text='# Original U-Net vs V89 comparison\n\n'
    text+='The original baseline is V66 original U-Net (100 completed training epochs, selected epoch37), checkpoint SHA25658a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382. This is the original prediction corresponding to historical reference20 new-root Dice0.8229. That historical score is not reused as a score under a different reference. Intermediate V85 or V88 predictions are NOT substituted for the original. The right-hand prediction is the exact published V89 output.\n\n'
    text+='Figures show original RGB, ORIGINAL U-Net (V66), OPTIMIZED output (V89), clearly labeled. Fig1 has overlay and white-background roots for all four dates. Fig2 uses the same matched native ROI in all four periods. Fig3 compares the three reported T2 regions, using identical native crops. All root contours, widths/lengths and original prediction files remain unchanged. T1 is identical. This export performs no training, inference, postprocessing or annotation modification.\n\n'
    text+='V89 is the inherited learned ConvLSTM/GAT features plus structural axis/junction decoding result; it is not a newly retrained U-Net. Its low-confidence state decisions and anatomical ownership limitations remain as reported in V89. Consistency-rule compliance is not independent accuracy. Optional metrics are read from already verified results under the SAME V87 reference, SAME fixed root foreground and SAME spatial scopes, with equal class supports and evaluated pixel counts asserted. T1 is excluded from three-class metrics. The original inner spatial audit is a development region and full-area agreement is in-sample descriptive agreement. No binary root-miss penalty is added.\n\n'
    text+='| Scope | Variant | Persistent | New | Extension | Macro |\n|---|---|---:|---:|---:|---:|\n'
    for scope,models in metrics.items():
        for name,m in models.items():text+='| '+scope+' | '+name+' | '+' | '.join(f"{r['Dice']:.6f}" for r in m['classes'])+f" | {m['macro_Dice']:.6f} |\n"
    text+='\nDownloads:\n\n'+''.join('- ['+r['file']+'](figures/'+r['file']+')\n' for r in manifest)+'\n- [Same-reference metrics CSV](Original_UNet_vs_V89_same_reference_metrics.csv)\n'
    (OUT/'RESULTS.md').write_text(text);(OUT/'source').mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    js(OUT/'completion.json',dict(state='complete',original_V66_vs_V89=True,no_training_or_data_edit=True));print('COMPARISON_COMPLETE',len(manifest),flush=True)


if __name__=='__main__':main()
