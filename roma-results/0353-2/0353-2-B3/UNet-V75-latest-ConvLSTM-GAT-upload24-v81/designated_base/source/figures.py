"""Verified original/joint comparison, raw100-epoch curves and honest three-class tables."""
import csv,json
import numpy as np
from PIL import Image
from prepare import ROOT,OUT,BASE,MODELS,PERIODS,js,sha,rows,csvout
from rendering import Canvas,FONT,PAL,isolated,overlay,cut
NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
VAR={'baseline':'Original','control':'Current-label control','convlstm_gat':'ConvLSTM + GAT'}
FIG=OUT/'figures'
def tables(records,outputs):
    for key,title in [('persistent','Persistent roots'),('newly_observed','Newly observed roots'),('tip_extension','Tip-extension segments')]:
        data=[r for r in records if r['scope']=='fixed_root_temporal' and r['name']==key]
        cv=Canvas(3300,1780,FIG);cv.text(1650,35,title+': same latest20 reference',65);cv.text(1650,135,'Fixed-root internal spatial holdout | Conditional on the frozen binary support',34)
        xx=[120,750,1630,2010,2390,2790];headers=['Model','Temporal branch','Precision','Recall','Dice','IoU']
        cv.d.line((90,235,3210,235),fill='black',width=6)
        for x,s in zip(xx,headers):cv.text(x,270,s,42,'lt')
        cv.d.line((90,340,3210,340),fill='black',width=3)
        for i,r in enumerate(data):
            y=390+i*130;vals=[NAMES[r['model']],VAR[r['variant']]]+[f"{float(r[k]):.4f}" for k in ['Precision','Recall','Dice','IoU']]
            for x,s in zip(xx,vals):cv.text(x,y,s,39,'lt')
            if i in [2,5]:cv.d.line((90,y+90,3210,y+90),fill='#B0B0B0',width=2)
        cv.d.line((90,1600,3210,1600),fill='black',width=6);cv.text(1650,1660,'Internal reference agreement; revised/model-assisted labels; not independent accuracy.',33);outputs.append(cv.save('Table_'+key+'_accuracy'))

def comparison_grid(stem,title,roi_by_period,outputs,mode='paired'):
    # Eight columns, each before/after shown at identical categorical scale.
    width=5840;height=260+len(roi_by_period)*1430+150;cv=Canvas(width,height,FIG);cv.text(width//2,30,title,70);cv.text(width//2,130,'Original vs ConvLSTM + GAT | Latest20 reference | Root geometry held fixed',36)
    xs=[180+j*700 for j in range(8)];ww=620
    labels=['RGB','Reference','U-Net original','U-Net joint','DeepLab original','DeepLab joint','SegFormer original','SegFormer joint']
    for i,(t,r) in enumerate(roi_by_period):
        top=240+i*1430;cv.text(20,top+650,t,52,'lt');raw=cut(RGB[t],r)
        for x,label in zip(xs,labels):cv.text(x+ww//2,top,label,37)
        cv.panel(raw,xs[0],top+400,ww,ww);rr=cut(REF[t],r);mm=cut(MASK[t],r)
        cv.panel(overlay(raw,rr),xs[1],top+75,ww,ww);cv.panel(isolated(rr,mm),xs[1],top+730,ww,ww,True)
        for j,m in enumerate(MODELS):
            for k,v in enumerate(['baseline','convlstm_gat']):
                a=cut(MAPS[m][v][t],r);x=xs[2+j*2+k];cv.panel(overlay(raw,a),x,top+75,ww,ww);cv.panel(isolated(a),x,top+730,ww,ww,True)
    cv.legend(height-110);outputs.append(cv.save(stem))

def curves(outputs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.text import Text
    fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
    curve_counts={}
    for i,m in enumerate(MODELS):
        hist={};zero={}
        for v in ['convlstm_gat','control']:
            p=OUT/(m+'_'+v);hist[v]=rows(p/'history.csv');zero[v]=json.loads((p/'epoch0_metrics.json').read_text());assert [int(r['epoch']) for r in hist[v]]==list(range(1,101));curve_counts[m+'_'+v]=100
        for v,style in [('convlstm_gat','-'),('control','--')]:
            hh=hist[v];z=zero[v];ep=list(range(101));tag='Joint' if v=='convlstm_gat' else 'Control'
            axes[i,0].plot(ep,[z['training_monitoring_CE']]+[float(r['training_monitoring_CE']) for r in hh],style,label=tag+' train',color='#0072B2',alpha=1 if v=='convlstm_gat' else .55)
            axes[i,0].plot(ep,[z['validation_monitoring_CE']]+[float(r['validation_pixel_CE']) for r in hh],style,label=tag+' validation',color='#D55E00',alpha=1 if v=='convlstm_gat' else .55)
            for k,field,lab,col in [(0,'validation_persistent_Dice','Persistent','#8856A7'),(1,'validation_new_Dice','New','#009E73'),(2,'validation_extension_Dice','Extension','#B47E00')]:
                axes[i,1].plot(ep,[z['validation']['classes'][k]['Dice']]+[float(r[field]) for r in hh],style,label=lab+' '+tag,color=col,alpha=1 if v=='convlstm_gat' else .5,linewidth=1.3)
            axes[i,2].plot([int(r['epoch']) for r in hh],[float(r['learning_rate']) for r in hh],style,label=tag,color='#0072B2' if v=='convlstm_gat' else '#555555')
        axes[i,0].set_title(NAMES[m]+': loss');axes[i,0].set_ylabel('Weighted pixel CE');axes[i,1].set_title(NAMES[m]+': validation categories');axes[i,1].set_ylim(0,1);axes[i,1].set_ylabel('Structured Dice');axes[i,2].set_title(NAMES[m]+': learning rate');axes[i,2].set_ylabel('Learning rate');axes[i,2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for ax in axes.ravel():ax.set_xlabel('Optimization epoch');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=7,ncol=2 if ax in axes[:,1] else 1)
    fig.canvas.draw()
    for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(FONT),size=tx.get_fontsize(),weight='bold'))
    stem='Fig_100_epochs_joint_and_same_label_control';fig.savefig(FIG/(stem+'_preview.png'),dpi=140,facecolor='white');tmp=FIG/'_curve600.png';fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
    p=FIG/(stem+'.tif')
    with Image.open(tmp) as im:im.convert('RGB').save(p,compression='tiff_adobe_deflate',dpi=(600,600))
    tmp.unlink();outputs.append(dict(file=p.name,sha256=sha(p)));return curve_counts

def main():
    global RGB,REF,MASK,MAPS
    ev=json.loads((OUT/'evaluation_verification.json').read_text());assert ev['passed'];FIG.mkdir(exist_ok=True);outputs=[]
    RGB={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')) for t in PERIODS};REF={t:np.array(Image.open(OUT/'data/reference'/f'{t}_classes.png')) for t in PERIODS};MASK={t:np.array(Image.open(OUT/'data/reference'/f'{t}_mask.png'))==255 for t in PERIODS}
    MAPS={m:{v:{t:np.array(Image.open(OUT/(m+'_'+v)/'full'/f'{t}_classes.png')) for t in PERIODS} for v in ['baseline','convlstm_gat']} for m in MODELS}
    rois=json.loads((OUT/'ROI_selection.json').read_text())
    cv=Canvas(6250,4200,FIG);cv.text(3125,30,'Temporal root classification across four periods',80);cv.text(3125,145,'Revised reference20 and three optimized temporal branches | Frozen root geometry',40)
    xs=[180,1400,2620,3840,5060];ww=1000;hh=436
    for x,label in zip(xs,['Original RGB','Reference20','U-Net joint','DeepLabV3+ joint','SegFormer-B0 joint']):cv.text(x+ww//2,245,label,42)
    for i,t in enumerate(PERIODS):
        y=310+i*925;cv.text(70,y+430,t,58);cv.panel(RGB[t],xs[0],y+220,ww,hh)
        arrays=[(REF[t],MASK[t])]+[(MAPS[m]['convlstm_gat'][t],None) for m in MODELS]
        for x,(a,mask) in zip(xs[1:],arrays):cv.panel(overlay(RGB[t],a),x,y,ww,hh);cv.panel(isolated(a,mask),x,y+hh+15,ww,hh,True)
        x0,y0,x1,y1=rois['matched_ROI'];cv.d.rectangle((xs[0]+round(x0/2541*ww),y+220+round(y0/1108*hh),xs[0]+round(x1/2541*ww),y+220+round(y1/1108*hh)),outline='#B88A00',width=5)
    cv.legend(4090);outputs.append(cv.save('Fig1_Four_period_optimized_temporal_comparison'))
    comparison_grid('Fig2_Four_period_matched_before_after','Matched temporal windows: original and optimized',[(t,rois['matched_ROI']) for t in PERIODS],outputs)
    comparison_grid('Fig3_Extension_before_after','True tip-extension detail: reference-selected windows',[(r['period'],r['ROI']) for r in rois['extension_ROIs']],outputs)
    for i,r in enumerate(rois['high_difference_windows'],1):comparison_grid('Fig4_High_difference_window_'+str(i),'High-disagreement window '+str(i)+': gains and regressions retained',[(t,r['ROI']) for t in PERIODS],outputs)
    records=rows(OUT/'Three_classes_accuracy_before_after.csv');tables(records,outputs);curve_counts=curves(outputs)
    summary=[]
    for m in MODELS:
        for v in ['baseline','control','convlstm_gat']:
            r=[a for a in records if a['model']==m and a['variant']==v and a['scope']=='fixed_root_temporal'];d={a['name']:float(a['Dice']) for a in r};overall=[a for a in records if a['model']==m and a['variant']==v and a['scope']=='including_binary_misses'];summary.append(dict(model=m,variant=v,persistent_Dice=d['persistent'],new_Dice=d['newly_observed'],extension_Dice=d['tip_extension'],macro_Dice=float(np.mean(list(d.values()))),macro_Dice_including_binary_misses=float(np.mean([float(a['Dice']) for a in overall]))))
    csvout(OUT/'Three_model_macro_summary.csv',summary)
    text='''# ConvLSTM + GAT temporal optimization against latest20\n\nAll six current-label modules completed100 real epochs: three joint ConvLSTM+GAT\nbranches and three matched controls without these modules. New temporal, memory\nand graph weights started randomly. Frozen original CNN/root-mask predictions remain\ninputs: this is initialization of new temporal modules, not full backbone retraining.\nThe old baseline was trained on18; it is reevaluated on20 here and its actual historical\nfigures and100-epoch logs are preserved under baseline_original, not retrospectively\nrewritten. No input label, root width, length, binary contour or original weight changed.\n\n## Conditional temporal Dice\n\n| Model | Variant | Persistent | New | Extension | Macro |\n|---|---|---:|---:|---:|---:|\n'''
    for r in summary:text+='| '+NAMES[r['model']]+' | '+VAR[r['variant']]+' | '+' | '.join(f"{r[k]:.4f}" for k in ['persistent_Dice','new_Dice','extension_Dice','macro_Dice'])+' |\n'
    text+='''\nThe U-Net joint model improves extension recognition versus both the original and\nmatched control, but reduces new/persistent Dice and does not exceed the matched\ncontrol macro score. Report all classes; do not combine flat auxiliary scores with\nstructured final scores. No separate ConvLSTM-only or GAT-only arm was run, so their\nindividual causal contribution is not established. Other model outcomes are retained\nin the table whether they improve or decline.\n\n## Scope and reading the outputs\n\nLatest20 labels include prior model-assisted revision and are not independently\nreviewed across all pixels. All four dates are from one B3 position, assigned to one\nouter partition. Fixed internal spatial masks provide development comparisons only,\nnot independent-tube/generalization accuracy. Conditional metrics score temporal\nclasses on the existing segmented roots. including_binary_misses additionally counts\nreference roots missed by the frozen binary segmentation. The two scopes must not\nbe mixed. T1 is baseline, not a trained change class.\n\nGraph targets come only from known user-confirmed identities, excluding unowned\nnew pixels and stale ownership; temporal class targets use20 throughout. Predicted\nconnections never receive manual IDs. Causal ConvLSTM/GAT preflight passed: future\nT4 perturbations do not change earlier representations, and both modules have real\ntraining gradients. Native skeleton chunks reconnect via learned affinities, endpoint\none-to-one matching and a junction-safe decoder. Predicted groups can still merge\nwrong identities or fragment a true root: Root_temporal_consistency.csv reports these\nerrors on known heldout root labels, not just the decoder's group-level color rule.\nA new-state group is uniformly new; an established group may have persistent shaft\nand growing tip. This biological decoder is explicit; it is not evidence that all learned\nconnections are correct. Same-date and cross-date connection metrics are separate\nfrom pixel Dice and appear in their own CSVs.\n\nAll optimized and control100-epoch loss curves use comparable train/validation\nweighted pixel CE, with total multi-objective training loss separately retained in CSV.\nRaw curves are not smoothed or generated. Checkpoints, optimizer/RNG resume states,\nselected epochs and validation-calibrated thresholds are preserved. The selected\ncheckpoint may precede epoch100; all100 actual epochs remain in history.\n\nAll isolated roots have white backgrounds. Categorical panels use nearest-neighbor\nresizing without root widening; black bold Times New Roman is used throughout.\nMatched windows maximize reference class coverage, extension windows use reference\nextension components, and high-difference windows use prediction disagreement alone,\nnot favorable accuracy. These illustrative windows can include development pixels.\nGains and regressions are retained; full arrays and tables are the quantitative record.\n'''
    (OUT/'RESULTS.md').write_text(text)
    js(OUT/'figure_manifest.json',dict(passed=True,files=outputs,typography='Times New Roman Bold, all text black',dpi=600,curves=curve_counts,reference_sha256=ev['latest20_source_sha256'],native_categorical_resizing=True,all_binary_supports_unchanged=True,high_difference_selection_not_accuracy=True))
    (OUT/'DELIVERABLES.md').write_text('# Deliverables\n\n'+''.join('- [Figure '+r['file']+'](figures/'+r['file']+')\n' for r in outputs)+'\n- [Three-class precision/recall/Dice/IoU](Three_classes_accuracy_before_after.csv)\n- [Macro summary](Three_model_macro_summary.csv)\n- [Root identity/continuity diagnostics](Root_temporal_consistency.csv)\n- [Same-date and cross-date connection metrics](Connection_accuracy_within_and_across.csv)\n- [Historical original figures](baseline_preserved_figures_complete.json)\n- [Scope and limitations](RESULTS.md)\n')
    print(json.dumps(dict(figures=len(outputs),modules_100_epochs=len(curve_counts),state='verified'),indent=2),flush=True)
if __name__=='__main__':main()
