#!/usr/bin/env python3
"""Publication-format graphics: restrained palette, native image content, vector text."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
import protocol as p

TEAL='#00857D';DARK='#374151';GREY='#9CA3AF';RED='#D55E00';BLUE='#0072B2'
NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0',
 'unet_context384':'U-Net / larger context','unet_context384_tta':'U-Net / context + TTA',
 'unet_segformer_mean':'U-Net + SegFormer','refined_unet':'Pretrained U-Net',
 'refined_unet_tta':'Pretrained U-Net / TTA','refined_legacy_mean':'U-Net ensemble'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.linewidth':.6,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})

def read(path):return json.loads(Path(path).read_text())
def color_rgb(hexcode):return np.array(matplotlib.colors.to_rgb(hexcode))*255
def mask_rgb(mask,color=TEAL):
 rgb=np.full((*mask.shape,3),255,np.uint8);rgb[mask]=color_rgb(color).astype(np.uint8);return rgb
def save(fig,path,image=False):
 fig.savefig(path.with_suffix('.pdf'),dpi=600)
 fig.savefig(path.with_suffix('.png'),dpi=600)
 fig.savefig(path.with_suffix('.tif'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
 if not image:fig.savefig(path.with_suffix('.svg'))
 plt.close(fig)
def off(ax):ax.set_axis_off()
def panel(ax,text):ax.text(-.08,1.07,text,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--selection',default='selection.json');args=a.parse_args()
 data=p.Data(args.data);root=args.run;out=root/'paper_figures';out.mkdir(exist_ok=True)
 selection=read(root/args.selection);winner=selection['winner'];metrics=read(root/'all_metrics.json');lengths=read(root/'root_length_agreement.json');topology=read(root/'topology_boundary_metrics.json')
 products=root/'root_products';captions={}
 # Whole-field overview retains actual RGB and actual reference/prediction pixels.
 fig,axes=plt.subplots(4,3,figsize=(7.2,4.65),layout='constrained')
 for i,t in enumerate(p.PERIODS):
  rgb=data.rgb[t];gt=data.gt[t]==255;pred=np.array(Image.open(products/(t+'_cleaned_mask.png')))==255
  for j,image in enumerate([rgb,mask_rgb(gt,DARK),mask_rgb(pred)]):axes[i,j].imshow(image,interpolation='none');off(axes[i,j])
  axes[i,0].add_patch(Rectangle((2070,192),384,384,fill=False,ec='#E3B553',lw=.75))
  axes[i,0].text(-.025,.5,t,transform=axes[i,0].transAxes,ha='right',va='center',fontweight='bold',fontsize=8)
  for j in range(3):panel(axes[i,j],chr(97+i*3+j))
 for ax,title in zip(axes[0],['Original RGB','Human reference','Refined root mask']):ax.set_title(title,pad=15)
 save(fig,out/'Fig1_four_period_root_segmentation',True)
 captions['Fig1']='Four aligned upper RGB regions, unchanged human foreground masks, and refined root masks. The gold box locates the shared384x384 detail in Fig2. All whole-field outputs include training pixels and are qualitative; official metrics use isolated C3/C4 predictions. Green denotes root foreground, not a temporal class. Original RGB is unaltered. Display scaling does not add image detail.'
 # Fixed shared C4 ROI, all periods; never choose a different flattering ROI per time.
 fig,axes=plt.subplots(4,4,figsize=(6.4,6.4),layout='constrained')
 for i,t in enumerate(p.PERIODS):
  folder=root/'candidates'/winner/'test';g=np.array(Image.open(folder/(t+'_human.png')))[192:576,:384]==255;pr=np.array(Image.open(folder/(t+'_cleaned.png')))[192:576,:384]==255
  rgb=data.rgb[t][192:576,2070:2454];error=np.full_like(rgb,255);error[g&pr]=color_rgb('#707777');error[~g&pr]=color_rgb(RED);error[g&~pr]=color_rgb(BLUE)
  for j,image in enumerate([rgb,mask_rgb(g,DARK),mask_rgb(pr),error]):axes[i,j].imshow(image,interpolation='none');off(axes[i,j]);panel(axes[i,j],chr(97+i*4+j))
  axes[i,0].text(-.04,.5,t,transform=axes[i,0].transAxes,ha='right',va='center',fontweight='bold')
 for ax,title in zip(axes[0],['RGB detail','Human reference','Refined prediction','Pixel disagreement']):ax.set_title(title,pad=15)
 fig.legend(handles=[Patch(color='#707777',label='Agreement'),Patch(color=RED,label='False positive'),Patch(color=BLUE,label='False negative')],loc='outside lower center',ncol=3,frameon=False)
 save(fig,out/'Fig2_fixed_repeated_benchmark_details',True)
 captions['Fig2']='The same native384x384 C4 region at all four periods. Predictions are generated with RGB restricted to the448-pixel C4 column; no outside training/validation context. Orange and blue mark disagreement relative to the supplied brush masks, not independently adjudicated anatomical errors. Previously viewed C4 is a repeated within-position benchmark, not a fresh independent test.'
 # Four foreground metrics, every axis fixed0..1, models distinguished by labels not rainbow bars.
 refined=max(['refined_unet','refined_unet_tta'],key=lambda n:metrics[n]['val']['cleaned']['root']['Dice'])
 names=list(dict.fromkeys(['unet','deeplab','segformer',refined,winner]));yy=np.arange(len(names))
 fig,axes=plt.subplots(1,4,figsize=(7.2,2.8),sharey=True,layout='constrained')
 for ax,key in zip(axes,['Precision','Recall','Dice','IoU']):
  for i,n in enumerate(names):
   color=TEAL if n==winner else GREY;val=metrics[n]['val']['cleaned']['root'][key];test=metrics[n]['test']['cleaned']['root'][key]
   ax.plot([val,test],[i,i],color=color,lw=.9,alpha=.75);ax.scatter([val],[i],s=28,facecolors='white',edgecolors=color,lw=1,zorder=3);ax.scatter([test],[i],s=28,c=color,marker='D',lw=0,zorder=3)
  ax.set(xlim=(0,1),xticks=[0,.25,.5,.75,1],xlabel=key);ax.grid(axis='x',color='#E5E7EB',lw=.5);ax.set_axisbelow(True)
  if key in ['Dice','Recall','Precision']:ax.axvline(.9 if key=='Precision' else .95,c='#C2A7A0',ls=':',lw=.8)
 axes[0].set_yticks(yy,[NAMES[n] for n in names]);axes[0].invert_yaxis()
 fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=DARK,color='none',label='Validation'),Line2D([],[],marker='D',color='none',mfc=DARK,mec=DARK,label='Repeated C4 benchmark')],loc='outside lower center',ncol=2,frameon=False)
 for i,ax in enumerate(axes):panel(ax,chr(97+i))
 save(fig,out/'Fig3_foreground_model_comparison')
 captions['Fig3']='Foreground Precision, Recall, strict pixel Dice and IoU at decisions frozen on C3. Open circles: validation; filled diamonds: repeated C4 benchmark. Teal identifies the selected configuration. All axes span0..1. Dotted lines show pre-existing root-quality gates. Optimization budgets and objectives differ; these are trained configurations, not an equal-budget architecture ranking. All explored variants are retained in the tables.'
 # Human-mask-derived length agreement with a common scale across the two displayed configurations.
 with (root/'root_length_pairs.csv').open(encoding='utf-8-sig') as f:pairs=list(csv.DictReader(f))
 names2=['unet',winner] if winner!='unet' else ['unet',refined];fig,axes=plt.subplots(2,2,figsize=(6.2,5.5),layout='constrained')
 maxvalue=max(float(r[k]) for r in pairs if r['candidate'] in names2 and r['split']=='test' for k in ['human_length_px','predicted_length_px'])*1.06
 markers=['o','s','^','D']
 for j,n in enumerate(names2):
  rr=[r for r in pairs if r['candidate']==n and r['split']=='test'];x=np.array([float(r['human_length_px']) for r in rr]);y=np.array([float(r['predicted_length_px']) for r in rr]);m=lengths[n]['test'];color=TEAL if n==winner else '#737B84'
  for t,marker in zip(p.PERIODS,markers):
   hit=np.array([r['period']==t for r in rr]);axes[0,j].scatter(x[hit],y[hit],s=22,marker=marker,c=color,alpha=.85,lw=.3,edgecolors='white');axes[1,j].scatter(((x+y)/2)[hit],(y-x)[hit],s=22,marker=marker,c=color,alpha=.85,lw=.3,edgecolors='white')
  axes[0,j].plot([0,maxvalue],[0,maxvalue],c='#9CA3AF',lw=.8,ls='--');axes[0,j].set(xlim=(0,maxvalue),ylim=(0,maxvalue),xlabel='Human-mask skeleton length (px)',ylabel='Predicted skeleton length (px)',title=NAMES[n])
  axes[0,j].text(.04,.94,'R² = {:.3f}\nRMSE = {:.1f} px'.format(m['R2_agreement'],m['RMSE_px']),transform=axes[0,j].transAxes,va='top',fontsize=8)
  for key,ls in [('mean_bias_px','-'),('Bland_Altman_lower_px','--'),('Bland_Altman_upper_px','--')]:axes[1,j].axhline(m[key],c='#9CA3AF',lw=.8,ls=ls)
  axes[1,j].set(xlabel='Mean skeleton length (px)',ylabel='Prediction − reference (px)')
 lo=min(ax.get_ylim()[0] for ax in axes[1]);hi=max(ax.get_ylim()[1] for ax in axes[1])
 for ax in axes[1]:ax.set_ylim(lo,hi)
 for ax,label in zip(axes.ravel(),'abcd'):panel(ax,label)
 fig.legend(handles=[Line2D([],[],marker=m,color='none',mfc=DARK,mec='none',label=t) for t,m in zip(p.PERIODS,markers)],loc='outside lower center',ncol=4,frameon=False)
 save(fig,out/'Fig4_root_length_agreement')
 captions['Fig4']='Skeleton-length agreement and Bland–Altman diagrams for32 descriptive block-period pairs:8 nonoverlapping C4 blocks repeated4 times. Both configurations use their C3-selected cleanup; the previous raw U-Net baseline is separately reported in comparison_summary.csv. Common axes permit comparison. Reference length is derived from human brush masks, not independently measured manual length. Units are native pixels; no physical calibration is supplied. R²=1−SSE/SST; negative values are retained. Pairs are correlated, not32 biological replicates; no inferential significance is claimed.'
 # Score curves, plus actual cleaned operating point; sampled raw-score PR is kept separate.
 fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained');g=np.concatenate([data.gt[t][:,2070:2518].ravel() for t in p.PERIODS]);styles=['--',':','-.','-']
 for n,style in zip(list(dict.fromkeys(['unet','deeplab','segformer',winner])),styles):
  scores=np.concatenate([np.load(root/'candidates'/n/'test'/(t+'_probability.npy')).ravel() for t in p.PERIODS]);mm=[p.metrics(p.counts(g,scores>=th)) for th in np.linspace(.01,.99,99)]
  axes[0].plot([m['Recall'] for m in mm],[m['Precision'] for m in mm],c=TEAL if n==winner else '#939AA3',lw=1.5 if n==winner else .8,ls=style,label=NAMES[n])
 m=metrics[winner]['test']['cleaned']['root'];axes[0].scatter(m['Recall'],m['Precision'],marker='*',s=65,c=TEAL,zorder=5,label='Frozen refined decision')
 axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Root recall',ylabel='Root precision');axes[0].legend(frameon=False,fontsize=6)
 with (root/'unet_pretrained/history.csv').open(encoding='utf-8-sig') as f:hist=list(csv.DictReader(f))
 hh=[r for r in hist if r['validation_Dice']]
 for key,color,style in [('Dice',TEAL,'-'),('Recall','#737B84','--'),('Precision','#A7ADB5',':')]:axes[1].plot([int(r['epoch']) for r in hh],[float(r['validation_'+key]) for r in hh],c=color,ls=style,lw=1.1,label=key)
 axes[1].axvline(read(root/'unet_pretrained/training_run.json')['best_epoch'],c='#BBBBBB',ls=':',lw=.8);axes[1].set(xlabel='Epoch',ylabel='Validation score at threshold0.5',ylim=(0,1));axes[1].legend(frameon=False,fontsize=7)
 for ax,label in zip(axes,'ab'):panel(ax,label)
 save(fig,out/'Fig5_precision_recall_and_training')
 captions['Fig5']='Raw-score Precision–Recall curves on the repeated C4 benchmark and fixed-threshold C3 learning curves of the pretrained U-Net trial. Star: the actual frozen cleaned decision, which may differ from the raw threshold curve. The selected epoch uses validation only. Graphs show measured values, without synthetic confidence bands.'
 # Deterministic removal-rich validation detail, not a favorable test crop.
 best=(-1,None,None,None)
 for t in p.PERIODS:
  removed=np.array(Image.open(root/'candidates'/winner/'val'/(t+'_removed.png')))==255
  for y in list(range(0,917,192))+[916]:
   for x in [0,128,256]:
    n=int(removed[y:y+192,x:x+192].sum())
    if n>best[0]:best=(n,t,x,y)
 count,t,x,y=best;folder=root/'candidates'/winner/'val';sl=(slice(y,y+192),slice(x,x+192))
 raw=np.array(Image.open(folder/(t+'_raw.png')))[sl]==255;cl=np.array(Image.open(folder/(t+'_cleaned.png')))[sl]==255
 changes=np.full((192,192,3),255,np.uint8);changes[raw&cl]=color_rgb('#8A9199');changes[raw&~cl]=color_rgb(RED);changes[~raw&cl]=color_rgb(TEAL)
 images=[data.rgb[t][y:y+192,1558+x:1558+x+192],mask_rgb(raw,DARK),mask_rgb(cl),changes]
 fig,axes=plt.subplots(1,4,figsize=(7.2,2.05),layout='constrained')
 for ax,image,title,label in zip(axes,images,['RGB detail','Raw prediction','Refined prediction','Changed pixels'],'abcd'):ax.imshow(image,interpolation='none');off(ax);ax.set_title(title);panel(ax,label)
 fig.legend(handles=[Patch(color=RED,label='Removed'),Patch(color=TEAL,label='Added')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,out/'Fig6_refinement_detail',True)
 captions['Fig6']='The native192x192 C3 window with the greatest number of removed candidates, chosen using removal counts rather than test accuracy. Period '+t+', global origin('+str(1558+x)+','+str(y)+'). Changes include frozen threshold, hysteresis and compact-component filtering. Removed candidates are not all independently confirmed noise; true-reference pixels removed are quantified in noise_audit.csv. Original RGB and actual mask pixels are shown without root-width redrawing.'
 # Stage-specific metrics retain failures that a pooled score could hide.
 fig,axes=plt.subplots(1,4,figsize=(7.2,2.25),layout='constrained')
 for ax,key,label in zip(axes,['Precision','Recall','Dice','IoU'],'abcd'):
  for split,color,marker in [('val','#9CA3AF','o'),('test',TEAL,'D')]:
   ax.plot(range(4),[metrics[winner][split]['cleaned']['per_period'][t][key] for t in p.PERIODS],c=color,marker=marker,ms=3.5,lw=1,label='Validation' if split=='val' else 'Repeated C4')
  ax.set(xticks=range(4),xticklabels=p.PERIODS,ylim=(0,1),ylabel=key);panel(ax,label)
 axes[0].legend(frameon=False,fontsize=6);save(fig,out/'FigS1_period_specific_metrics')
 captions['FigS1']='Foreground scores at each period. All thresholds and refinement parameters are frozen from C3; test traces refer to previously viewed C4. These are segmentation-performance trajectories, not measurements of biological growth or mortality.'
 # Foreground-score display with a perceptually uniform palette and one common scale.
 fig,axes=plt.subplots(2,2,figsize=(7.2,3.7),layout='constrained')
 for ax,t,label in zip(axes.ravel(),p.PERIODS,'abcd'):
  score=np.array(Image.open(products/(t+'_root_score_u16.png'))).astype(np.float32)/65535
  im=ax.imshow(score,cmap='cividis',vmin=0,vmax=1,interpolation='none');off(ax);ax.set_title(t);panel(ax,label)
 fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,pad=.02,label='Root foreground score')
 save(fig,out/'FigS2_foreground_scores',True)
 captions['FigS2']='Whole-upper-field sigmoid/ensemble foreground scores on a common0..1 cividis scale. These are uncalibrated model scores, not guaranteed correctness probabilities. Whole-field images include training regions.'
 p.js(root/'figure_captions.json',captions)
 (root/'Figure_captions_EN.md').write_text('\n\n'.join('**'+name+'** '+text for name,text in captions.items())+'\n')
 p.js(root/'figure_style.json',dict(palette=dict(root=TEAL,reference=DARK,false_positive=RED,false_negative=BLUE),
  output_dpi=600,formats=['PNG','TIFF LZW','PDF','SVG for quantitative plots'],font='DejaVu Sans',PDF_fonttype=42,
  native_RGB_altered=False,mask_widths_redrawn=False,source_resolution_increased=False))
 print('PAPER_FIGURES_READY',len(captions),flush=True)
if __name__=='__main__':main()
