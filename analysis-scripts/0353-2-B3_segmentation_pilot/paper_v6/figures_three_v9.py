#!/usr/bin/env python3
"""Three-model manuscript graphics, fixed regions and unmodified image content."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import protocol as p
from figures import save,mask_rgb,DARK,TEAL,RED,BLUE
from evaluate_three_v9 import NAMES

def read(f):return json.loads(Path(f).read_text())
def mask(f):return np.array(Image.open(f))>0
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);args=a.parse_args();run=args.run;data=p.Data(args.data);metrics=read(run/'all_metrics.json');lengths=read(run/'root_length_agreement.json');out=run/'figures';out.mkdir(exist_ok=True);captions={};colors={n:c for n,c in zip(NAMES,[TEAL,BLUE,'#79509A'])}
 fig,axes=plt.subplots(4,5,figsize=(10.2,4.8),layout='constrained')
 for i,t in enumerate(p.PERIODS):
  images=[data.rgb[t],mask_rgb(data.gt[t]==255,DARK)]+[mask_rgb(mask(run/n/'images'/(t+'_root_mask.png')),colors[n]) for n in NAMES]
  for ax,im in zip(axes[i],images):ax.imshow(im,interpolation='none');ax.set_axis_off()
  axes[i,0].text(-.02,.5,t,transform=axes[i,0].transAxes,ha='right',fontweight='bold')
 for ax,label in zip(axes[0],['Original RGB','Human reference']+list(NAMES.values())):ax.set_title(label)
 save(fig,out/'Fig1_four_period_three_model_overview',True);captions['Fig1']='Four upper RGB fields and binary root segmentation by all three trained configurations. Whole-field predictions include training regions and are qualitative. RGB and human masks are unchanged. Colors identify models, not temporal root classes. Native outputs retain2541x1108 pixels; figure rendering does not add optical detail.'
 # One fixed ROI in C4, all dates and all models, selected before looking at errors.
 fig,axes=plt.subplots(4,5,figsize=(8.2,6.6),layout='constrained')
 for i,t in enumerate(p.PERIODS):
  sl=(slice(192,576),slice(0,384));g=data.gt[t][192:576,2070:2454]==255;images=[data.rgb[t][192:576,2070:2454],mask_rgb(g,DARK)]
  for n in NAMES:
   pred=mask(run/n/'evaluation/test'/(t+'_after.png'))[sl];im=np.full((*g.shape,3),255,np.uint8);im[g&pred]=[75,85,85];im[~g&pred]=[213,94,0];im[g&~pred]=[0,114,178];images.append(im)
  for ax,im in zip(axes[i],images):ax.imshow(im,interpolation='none');ax.set_axis_off()
  axes[i,0].text(-.04,.5,t,transform=axes[i,0].transAxes,ha='right',fontweight='bold')
 for ax,label in zip(axes[0],['Original RGB','Human reference']+list(NAMES.values())):ax.set_title(label)
 fig.legend(handles=[Patch(color='#4B5555',label='True positive'),Patch(color=RED,label='False positive'),Patch(color=BLUE,label='False negative')],loc='outside lower center',ncol=3,frameon=False)
 save(fig,out/'Fig2_fixed_C4_disagreement',True);captions['Fig2']='Fixed native384x384 region (whole-image x2070,y192), identical across dates/models. Gray: true-positive root pixels; orange: false positives; blue: missed annotated roots; white: true background. The entire C4 region remains in the formal metrics; no difficult validation/test areas are removed.'
 fig,axes=plt.subplots(2,4,figsize=(8.6,4.5),layout='constrained')
 for i,split in enumerate(['val','test']):
  for j,key in enumerate(['Precision','Recall','Dice','IoU']):
   ax=axes[i,j]
   for k,n in enumerate(NAMES):
    initial=metrics[n][split]['initial']['root'][key];value=metrics[n][split]['after']['root'][key];ax.plot([initial,value],[k,k],color=colors[n],lw=1);ax.scatter([initial],[k],s=24,facecolors='white',edgecolors=colors[n]);ax.scatter([value],[k],s=24,color=colors[n],marker='D')
   ax.set(xlim=(0,1),xticks=[0,.25,.5,.75,1],xlabel=key);ax.grid(axis='x',color='#E5E7EB',lw=.5);ax.set_axisbelow(True)
   if j==0:ax.set_yticks(range(3),list(NAMES.values()));ax.set_title('C3 validation' if split=='val' else 'Repeated C4 benchmark',loc='left')
   else:ax.set_yticks(range(3),['']*3)
   ax.invert_yaxis()
 fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=DARK,color='none',label='Starting network / raw mask'),Line2D([],[],marker='D',mfc=DARK,color='none',label='Validation-selected final')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,out/'Fig3_three_model_strict_metrics');captions['Fig3']='Strict pixel foreground metrics for the starting raw networks and validation-selected fine-tuning/postprocessing. The starting raw network is not the complete published v8 U-Net connection pipeline. All models receive8epochs x32 native256 training crops with matched data groups and objective; starting training histories, pretraining, inference context and optimizer differ, so this is not an equal-total-compute architecture benchmark. Model/threshold selection uses C3. C4 is a repeatedly inspected same-position benchmark, not an independent test.'
 with (run/'root_length_pairs.csv').open(encoding='utf-8-sig') as f:rr=list(csv.DictReader(f))
 markers=['o','s','^','D']
 for split in ['val','test']:
  fig,axes=plt.subplots(2,3,figsize=(8.6,5.7),layout='constrained');subset=[r for r in rr if r['partition']==split and r['mode']=='after'];maxv=max(float(r[k]) for r in subset for k in ['human_length_px','predicted_length_px'])*1.06
  for j,n in enumerate(NAMES):
   rows=[r for r in subset if r['model']==n];x=np.array([float(r['human_length_px']) for r in rows]);y=np.array([float(r['predicted_length_px']) for r in rows]);m=lengths[n][split]['after'];coef=np.polyfit(x,y,1);xx=np.array([0,maxv])
   for t,marker in zip(p.PERIODS,markers):
    hit=np.array([r['period']==t for r in rows]);axes[0,j].scatter(x[hit],y[hit],s=24,marker=marker,color=colors[n],edgecolors='white',lw=.3);axes[1,j].scatter(((x+y)/2)[hit],(y-x)[hit],s=24,marker=marker,color=colors[n],edgecolors='white',lw=.3)
   axes[0,j].plot(xx,xx,color='#9CA3AF',ls='--',lw=.8);axes[0,j].plot(xx,np.polyval(coef,xx),color=colors[n],lw=.9);axes[0,j].set(xlim=(0,maxv),ylim=(0,maxv),xlabel='Human-mask skeleton length (px)',ylabel='Prediction length (px)' if j==0 else '',title=NAMES[n]);axes[0,j].text(.04,.95,'R² (agreement) = %.3f\nRMSE = %.1f px'%(m['R2_agreement'],m['RMSE_px']),va='top',transform=axes[0,j].transAxes)
   for key,style in [('mean_bias_px','-'),('Bland_Altman_lower_px','--'),('Bland_Altman_upper_px','--')]:axes[1,j].axhline(m[key],color='#9CA3AF',ls=style,lw=.8)
   axes[1,j].set(xlabel='Mean length (px)',ylabel='Prediction − human (px)' if j==0 else '')
  lo=min(ax.get_ylim()[0] for ax in axes[1]);hi=max(ax.get_ylim()[1] for ax in axes[1])
  for ax in axes[1]:ax.set_ylim(lo,hi);ax.set_xlim(0,maxv)
  fig.legend(handles=[Line2D([],[],marker=m,color='none',mfc=DARK,mec='none',label=t) for t,m in zip(p.PERIODS,markers)],loc='outside lower center',ncol=4,frameon=False)
  name='Fig4_root_length_'+('C3_validation' if split=='val' else 'C4_benchmark');save(fig,out/name);captions[name]='All three models: length regression and Bland–Altman plots. Solid line is OLS; dashed diagonal is identity. Displayed R² is agreement1−SSE/SST, not squared Pearson correlation; negative values are retained. References are human-mask skeleton lengths in native pixels, not independent manual length measurements.32 correlated block-date pairs (8 blocks x4 periods); no significance or biological replication claim. Common plot scales across models.'
 fig,axes=plt.subplots(2,3,figsize=(8.6,5),layout='constrained')
 for j,n in enumerate(NAMES):
  with (run/n/'history.csv').open(encoding='utf-8-sig') as f:hist=list(csv.DictReader(f))
  train=[r for r in hist if r['loss']];valid=[r for r in hist if r['validation_Dice']];axes[0,j].plot([int(r['epoch']) for r in train],[float(r['loss']) for r in train],color=colors[n],marker='.',lw=1);axes[0,j].set(title=NAMES[n],xlabel='Fine-tuning epoch',ylabel='Training loss' if j==0 else '')
  axes[1,j].plot([int(r['epoch']) for r in valid],[float(r['validation_Dice']) for r in valid],color=colors[n],marker='o',ms=3,lw=1);axes[1,j].set(xlabel='Fine-tuning epoch',ylabel='Calibrated C3 Dice' if j==0 else '',ylim=(.5,.85));selected=read(run/n/'selection.json')['epoch'];axes[1,j].axvline(selected,color='#9CA3AF',ls=':',lw=.8)
 save(fig,out/'Fig5_actual_training_records');captions['Fig5']='Actual8epoch training records for all three models. C3 thresholds are calibrated at epoch0,2,4,6,8. Dotted line denotes retained epoch;0 means attempted fine-tuning did not surpass the starting network. Never interpret epoch0 retention as a newly improved network.'
 fig,axes=plt.subplots(2,3,figsize=(8.6,4.8),layout='constrained')
 for i,split in enumerate(['val','test']):
  for j,(key,label) in enumerate([('reference_pair_connectivity','Reference-pair connectivity'),('skeleton_endpoints','Skeleton endpoints'),('clDice_mean','Mean clDice')]):
   ax=axes[i,j]
   for k,n in enumerate(NAMES):
    before=metrics[n][split]['before']['topology'][key];after=metrics[n][split]['after']['topology'][key];ax.plot([before,after],[k,k],color=colors[n]);ax.scatter([before],[k],s=22,facecolors='white',edgecolors=colors[n]);ax.scatter([after],[k],s=22,color=colors[n],marker='D')
   ax.set_xlabel(label)
   if j==0:ax.set_yticks(range(3),list(NAMES.values()));ax.set_title('C3 validation' if split=='val' else 'Repeated C4 benchmark',loc='left')
   else:ax.set_yticks(range(3),['']*3)
   if key!='skeleton_endpoints':ax.set_xlim(0,1)
   ax.invert_yaxis();ax.grid(axis='x',color='#E5E7EB',lw=.5)
 fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=DARK,color='none',label='Selected network / raw mask'),Line2D([],[],marker='D',mfc=DARK,color='none',label='Final postprocessing')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,out/'Fig6_continuity_diagnostics');captions['Fig6']='Connection/cleanup diagnostics applied to each selected network. Reference-pair connectivity uses manual skeleton anchors8steps apart and1pixel anchor neighborhoods; this is a topology diagnostic, not a tolerant replacement for strict Dice or a biological root-instance count. Fewer endpoints alone do not demonstrate a correct connection. All added/removed reference-root and background pixel counts are in metrics.csv.'
 fig,axes=plt.subplots(1,2,figsize=(7.4,3.5),layout='constrained')
 for ax,split,col in zip(axes,['val','test'],[3,4]):
  for n in NAMES:
   positive=np.zeros(4096,np.int64);negative=np.zeros(4096,np.int64)
   for t in p.PERIODS:
    folder=run/n/'probabilities'/('val' if split=='val' else 'test/C4');pr=np.load(folder/(t+'_probability.npy'));gt=data.gt[t][:,p.COLS[col]:p.COLS[col]+448];bins=np.minimum(4095,(np.clip(pr,0,1)*4096).astype(np.int32));positive+=np.bincount(bins[gt==255],minlength=4096);negative+=np.bincount(bins[(gt!=255)&(gt!=128)],minlength=4096)
   tp=positive[::-1].cumsum();fp=negative[::-1].cumsum();recall=tp/max(1,positive.sum());precision=tp/np.maximum(1,tp+fp);ax.plot(recall,precision,color=colors[n],lw=1,label=NAMES[n])
   for stage,marker in [('before','o'),('after','D')]:
    q=metrics[n][split][stage]['root'];ax.scatter(q['Recall'],q['Precision'],s=24,marker=marker,facecolors='white' if stage=='before' else colors[n],edgecolors=colors[n],zorder=3)
  ax.set(xlim=(0,1),ylim=(0,1),xlabel='Root recall',ylabel='Root precision',title='C3 validation' if split=='val' else 'Repeated C4 benchmark');ax.grid(color='#E5E7EB',lw=.5)
 axes[0].legend(frameon=False,loc='lower left');fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec=DARK,color='none',label='Raw operating point'),Line2D([],[],marker='D',mfc=DARK,color='none',label='Delivered final mask')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,out/'Fig7_foreground_precision_recall');captions['Fig7']='Foreground precision-recall curves for the validation-selected networks, using4096 probability bins and every valid formal-column pixel. Curves are a binned display, not additional selection or independent evaluation. Open circles show frozen raw thresholds; filled diamonds show delivered connection/cleanup masks and need not lie on the raw-score curve. No background-dominated overall accuracy is substituted for root accuracy.'
 (run/'Figure_captions_EN.md').write_text('\n\n'.join('**'+k+'** '+v for k,v in captions.items())+'\n');print('V9_FIGURES_READY',len(captions),flush=True)
if __name__=='__main__':main()
