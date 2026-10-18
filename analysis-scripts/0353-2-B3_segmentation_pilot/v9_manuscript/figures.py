"""Publication layout with paired regions, exact metrics, fixed error fields."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
import common as c

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--tables',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--only');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);summary=c.read(a.tables/'summary.json');agg=summary['aggregate'];periods=c.csvread(a.tables/'per_period_metrics.csv');pairs=c.csvread(a.tables/'root_length_pairs.csv');caps=c.read(out/'captions.json') if (out/'captions.json').exists() else {}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,'axes.titlesize':8,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','legend.fontsize':6.8})
 def save(fig,name,caption,raster=False):
  if a.only and a.only!=name:plt.close(fig);return
  fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
  if not raster:fig.savefig(out/(name+'.svg'),bbox_inches='tight')
  with Image.open(out/(name+'.png')) as im:im.save(out/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
  plt.close(fig);caps[name]=caption;print('FIGURE_READY',name,flush=True)
 def pooled(n,region):return next(r for r in agg if r['model']==n and r['aggregation']==region+'_pooled4dates')
 # Main1: native layout, temporal grouping, and real image counts.
 fig=plt.figure(figsize=(7.2,5),layout='constrained');gs=fig.add_gridspec(2,1,height_ratios=[3,1.6]);ax=fig.add_subplot(gs[0]);im=np.array(Image.open(a.inputs/'upper/T1_RGB.png'));ax.imshow(im);groupcolors=['#56B4E9']*3+['#E69F00','#B18ABD']
 for i,x in enumerate(c.COLS):
  ax.add_patch(Rectangle((x,0),448,1108,fill=False,color=groupcolors[i],lw=1.5));ax.text(x+224,45,f'C{i}',ha='center',va='top',color='white',fontweight='bold',bbox=dict(facecolor='#222222',alpha=.7,edgecolor='none',pad=2))
 ax.set_axis_off();ax.set_title('(a) Native upper field: 2541 × 1108 pixels',loc='left')
 ax=fig.add_subplot(gs[1]);ax.set(xlim=(-.6,6.8),ylim=(3.8,-.8));ax.set_axis_off()
 for j,t in enumerate(c.PERIODS):
  ax.text(-.5,j,t,va='center')
  for i in range(5):ax.add_patch(Rectangle((i-.4,j-.33),.8,.66,facecolor=groupcolors[i],alpha=.85,edgecolor='white'));ax.text(i,j,f'C{i}',ha='center',va='center',fontsize=7)
 for i,label in enumerate(['Train','Train','Train','Validation','Benchmark']):ax.text(i,-.65,label,ha='center',fontsize=6.5)
 ax.text(4.65,.3,'4 original RGB images\n1 annotated location\n12 / 4 / 4 date-region views\n0 external test images',va='top',linespacing=1.6,fontsize=7.5)
 fig.suptitle('Spatial grouping is preserved across all four periods',fontsize=10)
 save(fig,'Fig1_spatial_grouping','Native T1 upper field with448px-wide columns: C0–C2 training, C3 validation and checkpoint/parameter selection, C4 repeatedly inspected same-position benchmark. Columns start at x22,534,1046,1558,2070;64px guards separate adjacent columns. The same column at every date remains in the same group.12/4/4 date-region views derive from only4 original RGB images at one location; there are0 independent external test images. Display colors indicate dataset roles, not root categories.',True)
 for region,name in [('C4','Fig2_C4_primary_segmentation_metrics'),('C3','FigS1_C3_validation_metrics')]:
  fig,axs=plt.subplots(1,4,figsize=(7.2,2.9),layout='constrained')
  for ax,key in zip(axs,c.METRICS):
   for i,n in enumerate(c.MODELS):
    v=pooled(n,region)[key];rr=[float(r[key]) for r in periods if r['model']==n and r['region']==region];ax.plot([0,v],[i,i],lw=2,color=c.COLORS[i],alpha=.4);ax.scatter(rr,i+np.array([-.14,-.045,.045,.14]),s=14,facecolors='white',edgecolors=c.COLORS[i],lw=.7,zorder=3);ax.scatter(v,i,s=35,marker='D',color=c.COLORS[i],zorder=4);ax.text(.98,i-.28,f'{v:.3f}',ha='right',color=c.COLORS[i],fontsize=7)
   ax.set(xlim=(0,1),ylim=(2.6,-.65),xticks=[0,.5,1],xlabel=key);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
   if key=='Precision':ax.set_yticks(range(3),[c.NAMES[n] for n in c.MODELS])
   else:ax.set_yticks([])
  fig.legend(handles=[Line2D([],[],marker='D',color='#444444',ls='',label='4-date pixel-count pooled'),Line2D([],[],marker='o',mfc='white',mec='#444444',ls='',label='Individual dates (descriptive)')],loc='outside lower center',ncol=2,frameon=False)
  fig.suptitle(region+(' · same-position benchmark (primary pilot evaluation)' if region=='C4' else ' · validation used for model selection'),fontsize=9)
  save(fig,name,'Strict foreground Precision, Recall, Dice and IoU. Diamonds pool TP/FP/FN across all four native region-date masks, counting each pixel once after window fusion. Hollow points show individual dates and are not independent biological replicates or confidence intervals. C4 is a repeatedly inspected same-position benchmark; C3 is selection validation. Both must be named explicitly, with no independent external-test claim. No value is altered or rounded before aggregation.')
 # Fixed native original-v9 field, unchanged for dates and models.
 fig,axs=plt.subplots(4,5,figsize=(7.2,6.1),layout='constrained');sl=(slice(192,576),slice(2070,2454));local=(slice(192,576),slice(0,384))
 for i,t in enumerate(c.PERIODS):
  rgb=np.array(Image.open(a.inputs/'upper'/f'{t}_RGB.png'))[sl];g=np.array(Image.open(a.inputs/'upper'/f'{t}_mask.png'))[sl]==255;human=np.full((*g.shape,3),255,np.uint8);human[g]=[38,47,53];images=[rgb,human]
  for n in c.MODELS:
   pred=np.array(Image.open(a.inputs/'predictions'/n/'C4'/f'{t}_mask.png'))[local]>0;view=np.full((*g.shape,3),255,np.uint8);view[g&pred]=[75,85,85];view[~g&pred]=[213,94,0];view[g&~pred]=[0,114,178];images.append(view)
  for ax,v in zip(axs[i],images):ax.imshow(v,interpolation='none');ax.set_xticks([]);ax.set_yticks([]);[sp.set_visible(False) for sp in ax.spines.values()]
  axs[i,0].set_ylabel(t,rotation=0,labelpad=8,fontweight='bold')
 for ax,label in zip(axs[0],['Original RGB','Human mask']+[c.NAMES[n] for n in c.MODELS]):ax.set_title(label,fontsize=7)
 fig.legend(handles=[Patch(color='#4B5555',label='Matched root'),Patch(color='#D55E00',label='False root'),Patch(color='#0072B2',label='Missed root')],loc='outside lower center',ncol=3,frameon=False);fig.suptitle('C4 · identical native field at every date and model',fontsize=9)
 save(fig,'Fig3_C4_fixed_field_errors','Fixed original-v9 native384×384 ROI at whole-upper x2070,y192; identical for all dates/models and not selected using new outcomes. RGB and manual contours are unchanged. Gray: true-positive root; vermilion: background falsely predicted as root; blue: missed annotated root; white: true-negative background. The full C4 column, including difficult parts outside this illustration, contributes to formal scores. Binary masks do not classify growth, persistence, new roots or death.',True)
 markers=['o','s','^','D']
 for region,name in [('C4','Fig4_C4_root_length_agreement'),('C3','FigS2_C3_root_length_agreement')]:
  selected=[r for r in pairs if r['region']==region];lim=max(float(r[k]) for r in selected for k in ['human_length_px','predicted_length_px'])*1.06;fig,axs=plt.subplots(2,3,figsize=(7.2,5.1),layout='constrained');allerrors=[]
  for j,n in enumerate(c.MODELS):
   rr=[r for r in selected if r['model']==n];x=np.array([float(r['human_length_px']) for r in rr]);y=np.array([float(r['predicted_length_px']) for r in rr]);m=next(r for r in summary['length_agreement'] if r['model']==n and r['region']==region);xx=np.array([0,lim])
   for t,marker in zip(c.PERIODS,markers):
    hit=np.array([r['period']==t for r in rr]);axs[0,j].scatter(x[hit],y[hit],s=19,marker=marker,color=c.COLORS[j],edgecolors='white',lw=.3);axs[1,j].scatter(((x+y)/2)[hit],(y-x)[hit],s=19,marker=marker,color=c.COLORS[j],edgecolors='white',lw=.3)
   axs[0,j].plot(xx,xx,'--',color='#888888',lw=.7);axs[0,j].plot(xx,np.polyval(np.polyfit(x,y,1),xx),color=c.COLORS[j],lw=.85);axs[0,j].set(xlim=(0,lim),ylim=(0,lim),xlabel='Human-mask length (px)',ylabel='Predicted length (px)' if j==0 else '',title=c.NAMES[n]);axs[0,j].text(.04,.95,f'R² agreement = {m["R2_agreement"]:.3f}\nRMSE = {m["RMSE_px"]:.1f} px\nCCC = {m["CCC"]:.3f}',transform=axs[0,j].transAxes,va='top',fontsize=6.5)
   for key,ls in [('bias_px','-'),('LoA_lower_px','--'),('LoA_upper_px','--')]:axs[1,j].axhline(m[key],ls=ls,color='#777777',lw=.75)
   axs[1,j].axhline(0,color='#BBBBBB',lw=.5);axs[1,j].set(xlim=(0,lim),xlabel='Mean length (px)',ylabel='Prediction − human (px)' if j==0 else '');allerrors.extend([*(y-x),m['LoA_lower_px'],m['LoA_upper_px']])
  err=max(abs(np.asarray(allerrors)))*1.1
  for ax in axs[1]:ax.set_ylim(-err,err)
  fig.legend(handles=[Line2D([],[],marker=m,color='none',mfc='#444444',label=t) for t,m in zip(c.PERIODS,markers)],loc='outside lower center',ncol=4,frameon=False);fig.suptitle(region+' · root-length agreement and descriptive Bland–Altman',fontsize=9)
  save(fig,name,'All three models,32 correlated block-date pairs per panel (8 nonoverlapping224×277px blocks repeated at4 dates). References are human-mask skeleton lengths, not separate manual root-length measurements; units are native pixels, with no physical scale. Dashed diagonal is identity; solid regression is OLS. Displayed R²=1−SSE/SST relative to identity, not squared correlation or OLS R². Bland–Altman lines show descriptive bias±1.96SD; no independent-sample confidence interval or significance claim. High length agreement does not establish high pixel Dice or correct temporal identity.')
 fig,axs=plt.subplots(1,3,figsize=(7.2,2.5),layout='constrained',gridspec_kw={'width_ratios':[3,3,2]})
 for ax,region in zip(axs[:2],['C3','C4']):
  mat=np.array([[next(float(r['Dice']) for r in periods if (r['model'],r['region'],r['period'])==(n,region,t)) for t in c.PERIODS] for n in c.MODELS]);im=ax.imshow(mat,vmin=0,vmax=1,cmap='cividis',aspect='auto')
  for y in range(3):
   for x in range(4):ax.text(x,y,f'{mat[y,x]:.3f}',ha='center',va='center',color='white' if mat[y,x]<.55 else '#111111',fontsize=7)
  ax.set_xticks(range(4),c.PERIODS);ax.set_yticks(range(3),[c.NAMES[n] for n in c.MODELS] if region=='C3' else ['']*3);ax.set_title(region+' date-specific Dice')
 den=np.array([[next(r['root_prevalence'] for r in summary['root_prevalence'] if r['region']==region and r['period']==t) for t in c.PERIODS] for region in ['C3','C4']]);axs[2].imshow(den,vmin=0,vmax=den.max(),cmap='Greys',aspect='auto')
 for y in range(2):
  for x in range(4):axs[2].text(x,y,f'{den[y,x]*100:.1f}%',ha='center',va='center',fontsize=6,color='white' if den[y,x]>.6*den.max() else '#222222')
 axs[2].set(xticks=range(4),xticklabels=c.PERIODS,yticks=[0,1],yticklabels=['C3','C4'],title='Human-root pixel fraction');fig.colorbar(im,ax=axs[:2],fraction=.025,pad=.02,label='Dice',ticks=[0,.5,1]);save(fig,'FigS3_period_heterogeneity','Every period and region is retained. Dice uses a fixed0–1 color scale. Human-root prevalence shows that C3 and C4 differ in root abundance, affecting precision and PR interpretation; this is descriptive, not causal attribution. Four periods are related repeated observations, not biological replicate groups.')
 methods=['C3_pooled4dates','C4_pooled4dates','C3_C4_arithmetic_region_mean','C3_C4_count_pooled8views','C3_C4_macro8dateviews'];labels=['C3 pooled 4 dates','C4 pooled 4 dates','Mean of C3 and C4','Pixel pooled C3 + C4','Mean of 8 date views'];fig,axs=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
 for ax,key in zip(axs,['Dice','IoU']):
  ax.axhspan(1.5,4.5,color='#EDEEF0',zorder=0)
  for j,n in enumerate(c.MODELS):
   vals=[next(r[key] for r in agg if r['model']==n and r['aggregation']==method) for method in methods];ax.scatter(vals,np.arange(5)+(j-1)*.18,s=23,color=c.COLORS[j],label=c.NAMES[n])
  ax.set(xlim=(0,1),ylim=(4.6,-.6),xlabel=key,xticks=[0,.25,.5,.75,1]);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.set_yticks(range(5),labels if key=='Dice' else ['']*5)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False);fig.suptitle('Aggregation methods answer different questions',fontsize=9);save(fig,'FigS4_aggregation_comparison','C3 and C4 must remain separately identified because C3 selected the model. Mixed-region arithmetic means, pooled counts and equal-weighted8-date-view means differ numerically. Gray rows combine selection validation and benchmark and are exploratory only: they are not recommended headline performance or an independent test. Pixel pooling sums TP,FP,FN before computing metrics; the arithmetic mean gives equal weight to two region scores. None creates additional independent images.')
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
 for ax,region in zip(axs,['C3','C4']):
  baseline=None
  for j,n in enumerate(c.MODELS):
   curve=np.load(a.tables/f'{n}_{region}_PR.npz');m=next(r for r in summary['precision_recall'] if r['model']==n and r['region']==region);baseline=m['prevalence'];ax.plot(curve['recall'],curve['precision'],lw=1,color=c.COLORS[j],label=f'{c.NAMES[n]}  AP={m["Average_precision_exact"]:.3f}');q=pooled(n,region);ax.scatter(q['Recall'],q['Precision'],marker='D',s=24,color=c.COLORS[j],edgecolors='white',lw=.4,zorder=4)
  ax.axhline(baseline,ls=':',color='#777777',lw=.8,label=f'Root prevalence {baseline:.3f}');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Root recall',ylabel='Root precision',title=region+(' · validation' if region=='C3' else ' · repeated benchmark'));ax.legend(frameon=False,loc='lower left',fontsize=6);ax.grid(alpha=.15)
 save(fig,'FigS5_precision_recall','Average precision is recomputed exactly from all native valid pixel scores with equal-score ties grouped; plotted curves are downsampled only for file size. Diamonds show the delivered postprocessed masks, which need not lie on the selected raw-network probability curve. Dotted prevalence is the random-ranking PR baseline. PR and AP depend on prevalence; C3/C4 are different spatial conditions and not independent root-tube replicates.')
 fig,axs=plt.subplots(2,4,figsize=(7.2,4.7),layout='constrained')
 for i,region in enumerate(['C3','C4']):
  for j,key in enumerate(c.METRICS):
   ax=axs[i,j]
   for k,n in enumerate(c.MODELS):
    mean=next(r[key] for r in agg if r['model']==n and r['aggregation']==region+'_period_macro_mean');sd=next(r[key] for r in agg if r['model']==n and r['aggregation']==region+'_period_macro_SD');ax.errorbar(mean,k,xerr=sd,fmt='o',color=c.COLORS[k],capsize=3,ms=4,lw=1)
   ax.set(xlim=(0,1),ylim=(2.5,-.5),xticks=[0,.5,1],xlabel=key);ax.grid(axis='x',alpha=.15)
   if j==0:ax.set_yticks(range(3),[c.NAMES[n] for n in c.MODELS]);ax.set_title(region+' · mean ± SD',loc='left')
   else:ax.set_yticks([])
 fig.suptitle('Four-period means and descriptive temporal variation',fontsize=9)
 save(fig,'FigS6_four_period_mean_SD','Equal-weighted means and sample standard deviations of the four period-specific metrics, calculated separately within C3 and C4. Error bars are descriptive temporal SD, not SE, confidence intervals, or variation among independent plants/tubes. These date means differ from the main pooled-count metrics. No cross-region mean replaces the C4 main evaluation.')
 c.js(out/'captions.json',caps);(out/'Figure_captions_EN.md').write_text('\n\n'.join(f'**{k}.** {v}' for k,v in caps.items())+'\n')
if __name__=='__main__':main()
