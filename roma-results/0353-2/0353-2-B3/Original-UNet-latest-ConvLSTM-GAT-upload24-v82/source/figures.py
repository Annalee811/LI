"""New24-trained U-Net versus local corrections, with explicit edit-map audit."""
import csv,json
import numpy as np
from PIL import Image
from prepare import OUT,BASE,ANCHOR,PERIODS,js,sha
from rendering import Canvas,isolated,overlay,cut,FONT
FIG=OUT/'figures'
def main():
 v=json.loads((OUT/'evaluation_verification.json').read_text());assert v['passed'];FIG.mkdir(exist_ok=True);figs=[];rgb={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')) for t in PERIODS};ref={t:np.array(Image.open(OUT/'data/reference'/f'{t}_classes.png')) for t in PERIODS};before={t:np.array(Image.open(OUT/'frozen_unet'/f'{t}_classes.png')) for t in PERIODS};after={t:np.array(Image.open(OUT/'full'/f'{t}_classes.png')) for t in PERIODS}
 cv=Canvas(4900,4200,FIG);cv.text(2450,35,'Original U-Net: local refinement with the latest ConvLSTM + GAT',75);cv.text(2450,145,'Original U-Net baseline | Revised annotation24 | Root geometry fixed',37);xs=[170,1350,2550,3750];ww=1000;hh=436
 for x,s in zip(xs,['RGB','Revised reference','Original U-Net','After local refinement']):cv.text(x+ww//2,245,s,39)
 for i,t in enumerate(PERIODS):
  y=310+i*925;cv.text(75,y+440,t,58);cv.panel(rgb[t],xs[0],y+220,ww,hh)
  for x,a in zip(xs[1:],[ref[t],before[t],after[t]]):cv.panel(overlay(rgb[t],a),x,y,ww,hh);cv.panel(isolated(a),x,y+hh+15,ww,hh,True)
 cv.legend(4090);figs.append(cv.save('Fig1_Four_period_U_Net_local_correction'))
 windows=json.loads((OUT/'ROI_selection.json').read_text())['windows']
 for j,item in enumerate(windows,1):
  r=item['ROI'];cv=Canvas(4600,4980,FIG);cv.text(2300,35,'Local correction window '+str(j)+': original body labels retained',70);cv.text(2300,140,'Reference | newly trained U-Net | local correction | actual edited pixels',38);xx=[200,1300,2400,3500];w=850
  for i,t in enumerate(PERIODS):
   top=270+i*1100;cv.text(60,top+450,t,55);raw=cut(rgb[t],r);rr=cut(ref[t],r);bb=cut(before[t],r);aa=cut(after[t],r)
   for x,s in zip(xx,['Revised reference','Original U-Net','After local refinement','Edited pixels']):cv.text(x+w//2,top,s,39)
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
 axes[0,0].plot([int(r['epoch']) for r in mainhist],[float(r['training_loss']) for r in mainhist],label='Training');axes[0,0].plot([int(r['epoch']) for r in mainhist],[float(r['validation_loss']) for r in mainhist],label='Validation');axes[0,0].set_title('Original U-Net: historical100 epochs');axes[0,0].set_ylabel('Original temporal objective')
 for k,s,col in [(1,'Persistent','#8856A7'),(2,'New','#009E73'),(3,'Extension','#B47E00')]:axes[0,1].plot([int(r['epoch']) for r in mainhist],[float(r['validation_'+str(k)+'_Dice']) for r in mainhist],label=s,color=col)
 axes[0,1].set_ylim(0,1);axes[0,1].set_ylabel('Validation Dice');axes[0,1].set_title('Original U-Net: historical validation Dice')
 ep=[0]+[int(r['epoch']) for r in localhist];axes[1,0].plot(ep,[zero['training_CE']]+[float(r['training_local_CE']) for r in localhist],label='Local training CE');axes[1,0].plot(ep,[zero['validation_CE']]+[float(r['validation_local_CE']) for r in localhist],label='Local validation CE');axes[1,0].set_ylabel('Candidate-pixel cross entropy');axes[1,0].set_title('ConvLSTM + GAT: local targets only')
 axes[1,1].plot([int(r['epoch']) for r in localhist],[int(r['validation_fixed']) for r in localhist],label='Errors corrected',color='#009E73');axes[1,1].plot([int(r['epoch']) for r in localhist],[int(r['validation_harmed']) for r in localhist],label='Correct pixels harmed',color='#D55E00');axes[1,1].set_ylabel('Validation pixels');axes[1,1].set_title('Conservative edit calibration')
 for ax in axes.ravel():ax.set_xlabel('Training epoch');ax.grid(alpha=.18);ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False)
 fig.canvas.draw()
 for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(FONT),size=tx.get_fontsize(),weight='bold'))
 stem='Fig3_Two_stages_real_100_epoch_curves';fig.savefig(FIG/(stem+'_preview.png'),dpi=160,facecolor='white');tmp=FIG/'_curves.png';fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
 with Image.open(tmp) as im:im.convert('RGB').save(FIG/(stem+'.tif'),compression='tiff_adobe_deflate',dpi=(600,600))
 tmp.unlink();figs.append(dict(file=stem+'.tif',sha256=sha(FIG/(stem+'.tif'))));metrics=json.loads((OUT/'metrics.json').read_text());edit=metrics['test_local_edit_stats']
 text='''# Baseline correction: original U-Net plus the latest learned local ConvLSTM/GAT

The immutable baseline is the ORIGINAL U-Net shown in V75 BEFORE, not the
V75 AFTER ConvLSTM/GAT prediction and not the newer V78 U-Net. Its checkpoint
is V66 selected37, SHA25658a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382.
Its newly-observed Dice was .8229386284 against reference20. The same unchanged
prediction scores .7861928012 against the revised24 reference. These are different
reference evaluations, not model changes. The earlier .9233386899 belongs to the
PERSISTENT class of the different V75 AFTER baseline under24, not newly observed.
That previous baseline choice was mistaken for the user's intended comparison;
its files remain historical and are not relabeled as the original U-Net.

Only this original U-Net is refined. Latest V81 source24-learned local ConvLSTM/GAT
selected13, SHA256ce29d142a03c5b4ee0bf0ca9ace14e033e5bada5e1b548c8fa24ea57769646bc,
is inherited in full and locally adapted for100 additional actual epochs. The
original U-Net model, binary masks, widths/lengths, T1 and every noncandidate class
stay unchanged. DeepLabV3+ and SegFormer are not trained or changed.

Allowed edits are existing junction extension pixels, short inconsistent color
islands, history-supported persistent bridges, and small three-arm new-lateral
attachment cores. Existing continuous axes are persistent; new lateral attachments
are newly observed. A visual crossing alone is not growth evidence. Long terminal
extension and major body segments are protected. No root-wide recoloring or global
pixel class optimization is performed. Low-confidence candidates retain the original.

All supervision uses source24, preserved byte-for-byte. Separately recorded weak
history/attachment evidence supplements local training where single-period labels
may be incomplete. These proposals are not independently confirmed human labels.
Earlier same-root presence is required for persistence; future visibility alone
cannot prove earlier persistence. Manual IDs are targets only, never prediction inputs.

Historical original100-epoch curves use their actual original18 training reference;
new local100-epoch curves use24. Both BEFORE and AFTER metrics in this comparison
use the same24 reference, frozen root scope and internal spatial holdout pixels.
All dates are one B3 development position; no independent-tube accuracy is claimed.
Validation calibrates edits; test follows selection without threshold tuning.
Illustrations are selected by edit counts, not favorable accuracy. Harms, unresolved
identity conflicts, exact pixel-change audits, checkpoints and resume state are retained.

'''
 text+='Heldout local corrections: '+json.dumps(edit)+'\n\nWhole-field temporal Dice:\n\n| Variant | Persistent | New | Extension | Macro |\n|---|---:|---:|---:|---:|\n'
 for variant,m in metrics['whole'].items():text+='| '+variant+' | '+' | '.join(f"{r['Dice']:.4f}" for r in m['classes'])+f" | {m['macro_Dice']:.4f} |\n"
 text+='\nPaired historical-reference audit (same original prediction and same fixed spatial/root scope):\n\n| Reference | Original new-root Dice | Corrected new-root Dice |\n|---|---:|---:|\n'
 for ref,pair in metrics['paired_reference_evaluations'].items():text+='| '+ref+' | '+f"{pair['before']['classes'][1]['Dice']:.4f}"+' | '+f"{pair['after']['classes'][1]['Dice']:.4f}"+' |\n'
 (OUT/'RESULTS.md').write_text(text);js(OUT/'figure_manifest.json',dict(passed=True,files=figs,black_bold_Times_New_Roman=True,dpi=600,binary_geometry_unchanged=True,changes_outside_candidates_zero=True));print('FIGURES_COMPLETE',len(figs),flush=True)
if __name__=='__main__':main()
