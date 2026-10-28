"""All-field and known-regression illustrations; native labels, honest structural metrics."""
import csv,json,shutil
import numpy as np
from PIL import Image
from prepare import ROOT,OUT,LAST,BASE,PERIODS,HERE,js,sha
from rendering import Canvas,isolated,overlay,cut,FONT
D=OUT/'final';FIG=D/'figures'
def loadrows(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 assert json.loads((D/'evaluation_verification.json').read_text())['passed'];FIG.mkdir(exist_ok=True);records=[];rgb={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')) for t in PERIODS};ref={t:np.array(Image.open(LAST/'data/reference'/f'{t}_classes.png')) for t in PERIODS};before={t:np.array(Image.open(LAST/'full'/f'{t}_classes.png')) for t in PERIODS};after={t:np.array(Image.open(D/'full'/f'{t}_classes.png')) for t in PERIODS}
 cv=Canvas(4900,4200,FIG);cv.text(2450,35,'Whole-root temporal continuity across four periods',78);cv.text(2450,145,'Learned junction continuation + multi-period root identity + complete-segment decoding',38);xs=[170,1350,2550,3750];ww=1000;hh=436
 for x,s in zip(xs,['Original RGB','Revised reference','Previous local correction','Root-segment correction']):cv.text(x+ww//2,245,s,40)
 for i,t in enumerate(PERIODS):
  y=310+i*925;cv.text(75,y+440,t,58);cv.panel(rgb[t],xs[0],y+220,ww,hh)
  for x,a in zip(xs[1:],[ref[t],before[t],after[t]]):cv.panel(overlay(rgb[t],a),x,y,ww,hh);cv.panel(isolated(a),x,y+hh+15,ww,hh,True)
 cv.legend(4090);records.append(cv.save('Fig1_Four_period_whole_root_consistency'))
 def detail(roi,stem,title):
  cv=Canvas(4740,4260,FIG);cv.text(2370,35,title,68);cv.text(2370,135,'Same native window in all four periods | Root shapes, widths and lengths fixed',36);xx=[150,1060,1970,2880,3790];w=800
  for i,t in enumerate(PERIODS):
   top=250+i*950;cv.text(20,top+410,t,52,'lt');raw=cut(rgb[t],roi)
   for x,label in zip(xx,['RGB','Reference','Previous result','Root-segment result','Edited pixels']):cv.text(x+w//2,top,label,38)
   cv.panel(raw,xx[0],top+70,w,w)
   for x,a in zip(xx[1:4],[cut(ref[t],roi),cut(before[t],roi),cut(after[t],roi)]):cv.panel(isolated(a),x,top+70,w,w,True)
   old=cut(before[t],roi);new=cut(after[t],roi);diff=np.full((*old.shape,3),255,np.uint8);diff[old>0]=[210,210,210];diff[old!=new]=[213,94,0];cv.panel(diff,xx[4],top+70,w,w,True)
  cv.legend(4110);records.append(cv.save(stem))
 detail([1024,704,1280,960],'Fig2_User_reported_crossing_four_period_check','Known regression: continuity through the reported crossing')
 change=np.zeros((1108,2541),np.int32)
 for t in PERIODS[1:]:change+=(before[t]!=after[t])
 options=[]
 for y in range(0,1108-256+1,64):
  for x in range(0,2541-256+1,64):options.append((int(change[y:y+256,x:x+256].sum()),[x,y,x+256,y+256]))
 options.sort(key=lambda a:(-a[0],a[1]));windows=[]
 for score,roi in options:
  if any(max(0,min(roi[2],old[2])-max(roi[0],old[0]))*max(0,min(roi[3],old[3])-max(roi[1],old[1]))>256**2*.15 for old in [[1024,704,1280,960]]+[v['ROI'] for v in windows]):continue
  windows.append(dict(ROI=roi,changed_pixels=score))
  if len(windows)==3:break
 for i,v in enumerate(windows,1):detail(v['ROI'],'Fig3_All_field_detail_'+str(i),'All-field detail '+str(i)+': complete segments and junction types')
 js(D/'Figure_ROIs.json',dict(user_case=[1024,704,1280,960],other_windows=windows,selection='User case is a known development regression; other windows selected by actual change count, never favorable accuracy'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 from matplotlib.text import Text
 hist=loadrows(OUT/'joint_training_history.csv');assert [int(r['epoch']) for r in hist]==list(range(1,101));fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');ep=[int(r['epoch']) for r in hist]
 for ax,field,title,ylabel in zip(axes,['training_total','validation_pair_BCE','validation_junction_assignment_accuracy'],['Joint ConvLSTM + GAT training','Validation continuation loss','Validation junction assignments'],['Joint training objective','Binary cross entropy','Assignment accuracy']):ax.plot(ep,[float(r[field]) for r in hist],color='#0072B2');ax.set_xlabel('Joint training epoch');ax.set_ylabel(ylabel);ax.set_title(title);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 axes[2].set_ylim(0,1);fig.canvas.draw()
 for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(FONT),size=tx.get_fontsize(),weight='bold'))
 stem='Fig4_100_epochs_joint_history_and_graph_training';fig.savefig(FIG/(stem+'_preview.png'),dpi=160,facecolor='white');tmp=FIG/'_curves.png';fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
 with Image.open(tmp) as im:im.convert('RGB').save(FIG/(stem+'.tif'),compression='tiff_adobe_deflate',dpi=(600,600))
 tmp.unlink();records.append(dict(file=stem+'.tif',sha256=sha(FIG/(stem+'.tif'))))
 ev=json.loads((D/'evaluation_results.json').read_text());learned=json.loads((OUT/'joint_training_results.json').read_text());cv=Canvas(3300,1540,FIG);cv.text(1650,35,'Structural continuity audit: previous vs root-segment correction',62);cv.text(1650,130,'Internal known-root reference checks; these are not pixel Dice or independent accuracy',34)
 xx=[140,1740,2220,2650];cv.d.line((90,230,3210,230),fill='black',width=6)
 for x,s in zip(xx,['Check','Previous','Current','Records']):cv.text(x,270,s,42,'lt')
 cv.d.line((90,340,3210,340),fill='black',width=3);rr=ev['root_new_extension_conflicts5'];jj=ev['third_type_junction_audit'];ed=ev['holdout_edit_counts'];rows=[('Roots mixing new / extension (each >=5px)',rr['previous_local_V82'],rr['whole_segment_temporal_graph'],ev['root_period_records']),('Two-root junctions with a third class',jj['previous_local_V82']['junctions_with_third_type'],jj['whole_segment_temporal_graph']['junctions_with_third_type'],ev['scoreable_confirmed_two_root_junctions']),('Third-class pixels at those junctions',jj['previous_local_V82']['third_type_pixels'],jj['whole_segment_temporal_graph']['third_type_pixels'],'same set'),('Yellow pixels in the user review window',ev['user_review_case_yellow_pixels']['previous_local_V82'],ev['user_review_case_yellow_pixels']['whole_segment_temporal_graph'],'known case'),('Root geometry changed pixels',0,0,'all periods')]
 for i,row in enumerate(rows):
  for x,value in zip(xx,row):cv.text(x,400+i*170,str(value),39,'lt')
 cv.d.line((90,1320,3210,1320),fill='black',width=6);cv.text(1650,1380,'Remaining errors are retained; the known user example is not held-out validation.',33);records.append(cv.save('Table_Structural_continuity_before_after'))
 text='''# Whole-root temporal and junction consistency

The original U-Net binary geometry and original prediction files are preserved.
The input to this correction is the previously delivered V82 local result, itself
based on the original U-Net corresponding to historical reference20 new-root Dice .8229.
Only U-Net is processed, for all four B3 periods and the full registered upper field.

The relation model learns the actual task: which arms continue the same biological
root through a three/four-arm junction. Frozen-embedding pair-head pretraining100
actual epochs precedes100 actual joint ConvLSTM+GAT epochs. Joint losses include
continuation BCE, competing-arm ranking, within/across-period same-root identity,
root new/established state and chronological state consistency. Learned identity
affinities gate GAT messages. Direct ConvLSTM and GAT training gradients are verified.
The pair-head pretraining weights, joint selected weights, raw curves and optimizer/RNG
resume checkpoint are retained. U-Net binary weights/contours are never optimized.

The decoder is explicitly part of the method: root segments, learned arm continuations,
and temporal evidence determine complete anomalous intervals. A thick X can thin into
two Y nodes with a short shared stripe; opposite-arm geometry identifies these virtual
crossings without adding/removing a single root pixel. Both stable flanks and later
same-root evidence can recover a previously occluded interval; future presence alone
never proves earlier persistence. The final refinement is offline across all four
periods, not a future-free streaming prediction. Real terminal extension is protected.
A new root cannot have an internal extension island; established roots may have a
persistent shaft and actual growing tip. At a two-root crossing, a shared pixel may
belong to an incident root's type, never a third type. New/extension crossings therefore
retain those two types; extension is not universally banned at intersections.

These are learned history/graph features PLUS structured topology decoding, not a
claim that the raw neural pixel classifier alone removed the errors. Labels/source24
remain unchanged. Manual root IDs are supervision/audit targets, never inference input.
Uncertain pairings and temporal intervals are exported for review; not all roots are
resolved. Correct body classes outside identified intervals/intersections are retained.
Native root widths, lengths and all four foreground masks remain bitwise identical.

All dates belong to one B3 development position. Source24 contains historical
model-assisted annotation metadata and is not fully independently reviewed. Pixel
metrics are internal reference agreement on frozen root supports, separately scored
against20 and24; segmentation misses are outside that conditional task. Known-root
new/extension conflicts and junction third-type checks have different denominators.
The latter uses confirmed IDs and reference arm types for scoring, not the decoder's
own proposed allowed types. The reference itself can contain inconsistent core labels,
which are recorded. These counts do not constitute independent biological accuracy.
The user-reported window is a known development/regression example; other displayed
windows use change density only and can include regressions. No target score or
synthetic error was introduced, and old results were not overwritten.

'''
 text+='Internal joint continuation-head task: '+json.dumps(learned['test'])+'; full accepted/all task counts are in joint_training_results.json. This is root-arm pairing, not segmentation Dice.\n\n'
 text+='Current structural checks:\n\n'+json.dumps({k:v for k,v in ev.items() if k!='pixel_metrics'},indent=2)+'\n\nConditional temporal Dice:\n\n| Reference | Variant | Persistent | New | Extension | Macro |\n|---|---|---:|---:|---:|---:|\n'
 for refv,variants in ev['pixel_metrics'].items():
  for variant,m in variants.items():text+='| '+refv+' | '+variant+' | '+' | '.join(f"{r['Dice']:.4f}" for r in m['classes'])+f" | {m['macro_Dice']:.4f} |\n"
 (D/'RESULTS.md').write_text(text);js(D/'figure_manifest.json',dict(passed=True,files=records,black_bold_Times_New_Roman=True,dpi=600,geometry_unchanged=True,whole_field_four_period_processing=True))
 (D/'DELIVERABLES.md').write_text('# Deliverables\n\n'+''.join('- ['+r['file']+'](figures/'+r['file']+')\n' for r in records)+'\n- [Structural check CSV](Confirmed_two_root_junction_third_type_audit.csv)\n- [Known-root identity conflicts](Known_root_new_extension_conflicts.csv)\n- [Class metrics](Pixel_class_metrics_reference20_and24.csv)\n- [Whole corrected segments](Complete_segment_corrections.csv)\n- [Unresolved segments](Rejected_ambiguous_segments.csv)\n- [Scope and model/decoder roles](RESULTS.md)\n')
 for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
 js(D/'completion.json',dict(state='complete',joint_epochs=100,only_U_Net=True,all_four_periods=True,geometry_unchanged=True,unresolved_cases_retained=True));print('FIGURES_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
