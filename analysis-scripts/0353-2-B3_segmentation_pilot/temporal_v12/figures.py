"""Consistent palettes, fixed metric scales, all three models and real held-out outputs."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import common as c

def readcsv(f):
    with Path(f).open(encoding='utf-8-sig') as h:return list(csv.DictReader(h))
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run;out=run/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    reports=json.loads((run/'all_metrics.json').read_text());stats=json.loads((run/'root_length_agreement.json').read_text());pairs=readcsv(run/'root_length_pairs.csv');caps={};colors=['#0072B2','#D55E00','#009E73']
    def save(fig,name,caption,raster=False):
        fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        if not raster:fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        with Image.open(out/(name+'.png')) as im:im.save(out/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
        plt.close(fig);caps[name]=caption;print('PAPER_FIGURE',name,flush=True)
    legend=[Patch(color=c.PALETTE[k]/255,label=label) for k,label in [(4,'T1 baseline'),(1,'Persistent'),(2,'Newly observed'),(3,'Tip growth')]]
    fig,axs=plt.subplots(3,4,figsize=(15.4,6.2),layout='constrained')
    for i,n in enumerate(c.MODELS):
        for j,t in enumerate(c.PERIODS):
            axs[i,j].imshow(Image.open(run/n/'four_period_final_fit'/f'{t}_temporal_on_white.png'));axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
            for sp in axs[i,j].spines.values():sp.set_visible(False)
            if i==0:axs[i,j].set_title(t+' baseline' if j==0 else c.PERIODS[j-1]+' → '+t)
            if j==0:axs[i,j].set_ylabel(c.MODEL_NAMES[n])
    fig.legend(handles=legend,loc='outside lower center',ncol=4,frameon=False);fig.suptitle('Root-only temporal maps · final heads fitted to all B3 upper annotations',fontsize=11)
    save(fig,'Fig1_four_period_three_model_final_fit','Final deployment heads fitted using all B3 upper annotations. These panels show fitted-position behavior, not held-out accuracy. Root support is exactly frozen v9 segmentation. T1 is baseline; T2/T3/T4 refer only to the immediately previous date. Missing root visibility does not establish death.',True)
    fig,axs=plt.subplots(1,4,figsize=(10,6),layout='constrained')
    t='T2';gt=np.array(Image.open(run/'unet/evaluation/validation'/f'{t}_human_target.png'));view=np.full((*gt.shape,3),255,np.uint8);valid=gt!=255;view[valid]=c.PALETTE[gt[valid]];view[~valid]=[190,190,190]
    axs[0].imshow(view);axs[0].set_title('Human labels')
    for ax,n in zip(axs[1:],c.MODELS):
        pred=np.array(Image.open(run/n/'evaluation/validation'/f'{t}_classes.png'));ax.imshow(c.PALETTE[pred]);ax.set_title(c.MODEL_NAMES[n])
    for ax in axs:ax.set_axis_off()
    fig.suptitle('Spatially held-out C3 · T1 → T2 · evaluation heads',fontsize=11);fig.legend(handles=legend[1:],loc='outside lower center',ncol=3,frameon=False)
    save(fig,'Fig2_spatial_validation_T2','C3 spatial validation visualization using evaluation heads fitted on C0–C2. C3 was used for checkpoint and class calibration selection. Unlabelled temporal root pixels are gray in human reference. Regions/date are shared across all model panels; this is not an external tube test.',True)
    for split in ['validation','repeated_benchmark']:
        fig,axs=plt.subplots(1,4,figsize=(12,3.2),layout='constrained')
        for ax,metric in zip(axs,['Precision','Recall','Dice','IoU']):
            for i,n in enumerate(c.MODELS):
                vals=[r[metric] or 0 for r in reports[n][split]['end_to_end']['classes']];ax.bar(np.arange(3)+(i-1)*.24,vals,.24,color=colors[i],label=c.MODEL_NAMES[n])
            ax.set_xticks(range(3),['Persistent','New','Growth']);ax.set_ylim(0,1);ax.set_title(metric);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        axs[0].legend(frameon=False,fontsize=6);fig.suptitle('End-to-end temporal segmentation · '+split.replace('_',' '),fontsize=11)
        save(fig,'Fig3_class_metrics_'+split,'Per-class strict native-pixel temporal metrics, including errors from frozen root detection. Fixed0–1 axes. No external independent labelled test images. C4 is a previously inspected same-position benchmark, not a fresh independent test.')
    fig,axs=plt.subplots(2,3,figsize=(11,6),layout='constrained')
    for j,n in enumerate(c.MODELS):
        hist=readcsv(run/n/'history.csv');axs[0,j].plot([int(r['epoch']) for r in hist],[float(r['loss']) for r in hist],color=colors[j]);axs[0,j].set_title(c.MODEL_NAMES[n]);axs[0,j].set_ylabel('Training loss');axs[0,j].set_xlabel('Epoch')
        h=[r for r in hist if r['validation_macro_Dice']];xs=[int(r['epoch']) for r in h]
        for field,label,style in [('validation_macro_Dice','End-to-end','-'),('validation_conditional_macro_Dice','Detected-root only','--')]:axs[1,j].plot(xs,[float(r[field]) for r in h],style,color=colors[j],label=label)
        initial=json.loads((run/n/'epoch0_metrics.json').read_text())['end_to_end']['macro_Dice'];axs[1,j].axhline(initial,color='#444444',ls=':',lw=.8,label='Epoch0 baseline')
        axs[1,j].set_ylim(0,1);axs[1,j].set_ylabel('C3 temporal macro Dice');axs[1,j].set_xlabel('Epoch');axs[1,j].legend(frameon=False,fontsize=6)
    save(fig,'Fig4_temporal_training_records','Actual three-head continuation records using inherited v11 class multipliers and adjustments. Dotted line marks the retained epoch0 baseline. Solid validation curves include missed/false roots; dashed curves measure classification only on correctly detected reference roots. Curves precede the subsequent C3-only skeleton-chain decoding and checkpoint selection; final tables include it. All backbones and root masks remained frozen. Separate all-label final refits are in final_history.csv.')
    fig,axs=plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    for ax,n in zip(axs,c.MODELS):
        cm=np.array(reports[n]['validation']['end_to_end']['confusion_matrix']);norm=cm/np.maximum(cm.sum(1,keepdims=True),1);ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{norm[i,j]:.2f}',ha='center',va='center',color='white' if norm[i,j]>.6 else '#222222',fontsize=7)
        ax.set_xticks(range(4),['Background','Persistent','New','Growth'],rotation=35,ha='right');ax.set_yticks(range(4),['Background','Persistent','New','Growth']);ax.set_title(c.MODEL_NAMES[n]);ax.set_xlabel('Predicted');ax.set_ylabel('Human label')
    save(fig,'Fig5_validation_confusion','Row-normalized C3 end-to-end confusion matrices. Predicted background on a reference root is a miss; a background pixel assigned a root class is a false positive. Untyped root reference pixels are excluded.')
    for split in ['validation','repeated_benchmark']:
        fig,axs=plt.subplots(4,3,figsize=(10.5,12),layout='constrained')
        for i,k in enumerate(['all_roots','1','2','3']):
            selected=[r for r in pairs if r['split']==split and r['root_class']==k];maxv=max(max(float(r['human_length_px']),float(r['predicted_length_px'])) for r in selected)*1.08 or 1
            for j,n in enumerate(c.MODELS):
                ax=axs[i,j];rr=[r for r in selected if r['model']==n];x=np.array([float(r['human_length_px']) for r in rr]);y=np.array([float(r['predicted_length_px']) for r in rr]);ax.scatter(x,y,s=12,alpha=.75,color=colors[j],edgecolors='none');xx=np.array([0,maxv]);ax.plot(xx,xx,'--',color='#777777',lw=.8)
                if x.var():ax.plot(xx,np.polyval(np.polyfit(x,y,1),xx),color=colors[j],lw=.8)
                m=stats[n][split+'_'+k];r2='not defined' if m['R2_agreement'] is None else f'{m["R2_agreement"]:.3f}';ax.text(.04,.95,f'R² agreement = {r2}\nRMSE = {m["RMSE_px"]:.1f} px',va='top',transform=ax.transAxes,fontsize=7)
                ax.set_xlim(0,maxv);ax.set_ylim(0,maxv);ax.set_xlabel('Human-mask length (px)');ax.set_ylabel('Predicted length (px)');ax.set_title(c.MODEL_NAMES[n]+' · '+('All roots' if k=='all_roots' else c.NAMES[int(k)].replace('_',' ')),fontsize=8)
        save(fig,'Fig6_length_agreement_'+split,'All three models, total-root and class-specific length agreement. R²=1−SSE/SST relative to the identity line; negative values retained. Total-root length uses frozen v9 masks. References derive from human mask skeletons, not separate root-length measurements. Eight correlated spatial blocks repeated across dates;32 total-root pairs or24 class-date pairs; no biological independence or significance claim.')
    fig,axs=plt.subplots(3,3,figsize=(5.6,12),layout='constrained')
    for i,n in enumerate(c.MODELS):
        for j,t in enumerate(c.PERIODS[1:]):
            ax=axs[i,j];pr=np.array(Image.open(run/n/'evaluation/validation'/f'{t}_classes.png'));ax.imshow(c.PALETTE[pr]);ax.set_axis_off()
            if i==0:ax.set_title(c.PERIODS[j]+' → '+t)
            if j==0:ax.text(-.06,.5,c.MODEL_NAMES[n],transform=ax.transAxes,rotation=90,va='center')
    fig.suptitle('Held-out C3 temporal predictions · all transitions',fontsize=11);fig.legend(handles=legend[1:],loc='outside lower center',ncol=3,frameon=False)
    save(fig,'Fig7_all_transitions_validation','All three adjacent transitions on the same C3 validation column. All predictions use previous and current predicted-root-only inputs; no preceding manual root masks at inference.',True)
    fig,axs=plt.subplots(1,3,figsize=(8.3,6),layout='constrained')
    for ax,n in zip(axs,c.MODELS):
        ax.imshow(Image.open(run/n/'evaluation/validation/T2_error_map.png'));ax.set_title(c.MODEL_NAMES[n]);ax.set_axis_off()
    error_legend=[Patch(color=col,label=label) for col,label in [('#009E73','Correct root class'),('#D55E00','Missed root'),('#CC79A7','Wrong root class'),('#56B4E9','Background false root'),('#AAAAAA','Untyped / excluded')]]
    fig.legend(handles=error_legend,loc='outside lower center',ncol=3,frameon=False,fontsize=7);fig.suptitle('C3 validation · T1 → T2 · error localization',fontsize=11)
    save(fig,'Fig8_validation_error_localization','Same C3/T2 field for all three models; green means correct temporal root class, vermilion means missed reference root, magenta means root detected but assigned wrong temporal class, blue means background classified as a root, and gray marks excluded temporal reference pixels. White is correctly predicted background. Native error maps for every scored transition are retained.',True)
    c.js(out/'captions.json',caps);(out/'Figure_captions_EN.md').write_text('\n\n'.join(f'{k}: {v}' for k,v in caps.items())+'\n')
if __name__=='__main__':main()
