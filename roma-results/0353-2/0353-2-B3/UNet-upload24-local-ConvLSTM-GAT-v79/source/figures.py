"""New24-trained U-Net versus local corrections, with explicit edit-map audit."""
import csv,json
import numpy as np
from PIL import Image
from prepare import OUT,BASE,ANCHOR,PERIODS,js,sha
from rendering import Canvas,isolated,overlay,cut,FONT
FIG=OUT/'figures'
def main():
 v=json.loads((OUT/'evaluation_verification.json').read_text());assert v['passed'];FIG.mkdir(exist_ok=True);figs=[];rgb={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')) for t in PERIODS};ref={t:np.array(Image.open(OUT/'data/reference'/f'{t}_classes.png')) for t in PERIODS};before={t:np.array(Image.open(OUT/'frozen_unet'/f'{t}_classes.png')) for t in PERIODS};after={t:np.array(Image.open(OUT/'full'/f'{t}_classes.png')) for t in PERIODS}
 cv=Canvas(4900,4200,FIG);cv.text(2450,35,'U-Net temporal labels: local ConvLSTM + GAT correction',75);cv.text(2450,145,'Source24-trained U-Net frozen before local correction | No edits outside allowed candidates',37);xs=[170,1350,2550,3750];ww=1000;hh=436
 for x,s in zip(xs,['RGB','Reference24','Newly trained U-Net','U-Net + local ConvLSTM/GAT']):cv.text(x+ww//2,245,s,39)
 for i,t in enumerate(PERIODS):
  y=310+i*925;cv.text(75,y+440,t,58);cv.panel(rgb[t],xs[0],y+220,ww,hh)
  for x,a in zip(xs[1:],[ref[t],before[t],after[t]]):cv.panel(overlay(rgb[t],a),x,y,ww,hh);cv.panel(isolated(a),x,y+hh+15,ww,hh,True)
 cv.legend(4090);figs.append(cv.save('Fig1_Four_period_U_Net_local_correction'))
 windows=json.loads((OUT/'ROI_selection.json').read_text())['windows']
 for j,item in enumerate(windows,1):
  r=item['ROI'];cv=Canvas(4600,4980,FIG);cv.text(2300,35,'Local correction window '+str(j)+': original body labels retained',70);cv.text(2300,140,'Reference | newly trained U-Net | local correction | actual edited pixels',38);xx=[200,1300,2400,3500];w=850
  for i,t in enumerate(PERIODS):
   top=270+i*1100;cv.text(60,top+450,t,55);raw=cut(rgb[t],r);rr=cut(ref[t],r);bb=cut(before[t],r);aa=cut(after[t],r)
   for x,s in zip(xx,['Reference24','New U-Net','Local ConvLSTM + GAT','Edited pixels']):cv.text(x+w//2,top,s,39)
   for x,a in zip(xx[:3],[rr,bb,aa]):cv.panel(isolated(a),x,top+90,w,w,True)
   changed=bb!=aa;im=np.full((*changed.shape,3),255,np.uint8);im[bb>0]=[205,205,205];im[changed]=[213,94,0];cv.panel(im,xx[3],top+90,w,w,True)
  cv.legend(4810);figs.append(cv.save('Fig2_Local_connection_and_jump_window_'+str(j)))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 from matplotlib.text import Text
 def read(p):
  with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
 mainhist=read(ANCHOR/'unet/history.csv');localhist=read(OUT/'history.csv');assert [int(r['epoch']) for r in mainhist]==list(range(101));assert [int(r['epoch']) for r in localhist]==list(range(1,101));zero=json.loads((OUT/'epoch0_metrics.json').read_text());fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 axes[0,0].plot([int(r['epoch']) for r in mainhist],[float(r['training_loss']) for r in mainhist],label='Training');axes[0,0].plot([int(r['epoch']) for r in mainhist],[float(r['validation_loss']) for r in mainhist],label='Validation');axes[0,0].set_title('U-Net temporal-head training on24');axes[0,0].set_ylabel('Temporal training objective')
 for k,s,col in [(1,'Persistent','#8856A7'),(2,'New','#009E73'),(3,'Extension','#B47E00')]:axes[0,1].plot([int(r['epoch']) for r in mainhist],[float(r['validation_'+str(k)+'_Dice']) for r in mainhist],label=s,color=col)
 axes[0,1].set_ylim(0,1);axes[0,1].set_ylabel('Validation Dice');axes[0,1].set_title('New U-Net: three temporal categories')
 ep=[0]+[int(r['epoch']) for r in localhist];axes[1,0].plot(ep,[zero['training_CE']]+[float(r['training_local_CE']) for r in localhist],label='Local training CE');axes[1,0].plot(ep,[zero['validation_CE']]+[float(r['validation_local_CE']) for r in localhist],label='Local validation CE');axes[1,0].set_ylabel('Candidate-pixel cross entropy');axes[1,0].set_title('ConvLSTM + GAT: local targets only')
 axes[1,1].plot([int(r['epoch']) for r in localhist],[int(r['validation_fixed']) for r in localhist],label='Errors corrected',color='#009E73');axes[1,1].plot([int(r['epoch']) for r in localhist],[int(r['validation_harmed']) for r in localhist],label='Correct pixels harmed',color='#D55E00');axes[1,1].set_ylabel('Validation pixels');axes[1,1].set_title('Conservative edit calibration')
 for ax in axes.ravel():ax.set_xlabel('Training epoch');ax.grid(alpha=.18);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
 fig.canvas.draw()
 for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(FONT),size=tx.get_fontsize(),weight='bold'))
 stem='Fig3_Two_stages_real_100_epoch_curves';fig.savefig(FIG/(stem+'_preview.png'),dpi=160,facecolor='white');tmp=FIG/'_curves.png';fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
 with Image.open(tmp) as im:im.convert('RGB').save(FIG/(stem+'.tif'),compression='tiff_adobe_deflate',dpi=(600,600))
 tmp.unlink();figs.append(dict(file=stem+'.tif',sha256=sha(FIG/(stem+'.tif'))));metrics=json.loads((OUT/'metrics.json').read_text());edit=metrics['test_local_edit_stats']
 text='''# Source24 U-Net retraining and local-only ConvLSTM/GAT correction\n\nStage1 trains a new randomly initialized U-Net temporal CNN head for100 actual\nepochs, using the user-uploaded24 temporal annotation. Binary U-Net masks, weights,\nwidths and root lengths remain frozen; the old temporal CNN weights are not loaded.\nThe prediction-only25-channel RGB/binary cache is reused exactly; it contains no\nold temporal model output.\n\nStage2 freezes the newly trained U-Net temporal maps and trains a fresh causal\nConvLSTM/GAT local corrector for100 real epochs. Its pixel classification loss uses\nonly candidate junction/short-island pixels, with auxiliary same-root link, confirmed\nparent attachment and clean-root state targets from24. No global pixel task is\noptimized. Candidate masks use only frozen U-Net predictions and geometry, never\nreference classes or IDs. The inherited global pixel head is frozen and unused.\n\nAllowed edits are existing extension-colored pixels within4px of a predicted\nskeleton junction, and very short green/yellow color islands (at most16 skeleton\npixels, at most72 root pixels) with same-native-path opposing-color evidence, short internal
extension/persistent bridge anomalies with earlier root support, and2px three-arm
attachment cores near newly observed lateral colors. Long\nterminal extensions and major segments are not candidates. A visual crossing alone\nis not evidence of a true branch. Local learned class/edit scores decide whether\nto change a candidate; proposed extension at connection candidates is not applied.\nThe original label remains when confidence is insufficient. No forced root-wide\nmajority coloring or global new/established relabeling is performed.\n\nHistory-supported persistent bridges and confirmed-parent new-lateral birth points
provide separately recorded weak local supervision. Earlier same-root presence is
required for persistence; future visibility alone cannot establish earlier persistence.
A newly appearing lateral attachment is supervised as newly observed, rather than
copying its persistent mother's class. Conflicting single-date labels are downweighted
only for these high-evidence local targets, while source24 remains byte-preserved.
These derived targets/proposals are not presented as independent human ground truth;
Temporal_evidence_proposals.csv records them. No direct reference-ID input is used
at prediction. Primary accuracy metrics continue to use the unchanged manual labels.

The conservative edit threshold is chosen only on the fixed inner validation mask\nto reward corrected errors and penalize damage to already-correct labels. All test\nscoring follows selection. Every noncandidate pixel is asserted bitwise unchanged\nfrom the newly trained U-Net; T1 and every binary mask are also identical.\n\nAll four dates are one B3 development position. Revised/model-assisted annotation\nmetadata is retained; known ownership/class ambiguities are not silently edited.\nScores are internal reference agreement, not independent-tube generalization. Local\nedit precision, whole-field three-class Dice and root identity conflicts are distinct\nquantities. Remaining errors outside the allowed region are reported rather than\nchanged. Illustrations are selected by edit counts, not favorable accuracy.\n\n'''
 text+='Heldout local corrections: '+json.dumps(edit)+'\n\nWhole-field temporal Dice:\n\n| Variant | Persistent | New | Extension | Macro |\n|---|---:|---:|---:|---:|\n'
 for variant,m in metrics['whole'].items():text+='| '+variant+' | '+' | '.join(f"{r['Dice']:.4f}" for r in m['classes'])+f" | {m['macro_Dice']:.4f} |\n"
 (OUT/'RESULTS.md').write_text(text);js(OUT/'figure_manifest.json',dict(passed=True,files=figs,black_bold_Times_New_Roman=True,dpi=600,binary_geometry_unchanged=True,changes_outside_candidates_zero=True));print('FIGURES_COMPLETE',len(figs),flush=True)
if __name__=='__main__':main()
