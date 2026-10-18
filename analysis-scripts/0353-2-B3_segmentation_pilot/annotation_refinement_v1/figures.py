"""English publication figures: 600dpi TIFF/PNG plus PDF and editable vector charts."""
import argparse,re
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D
import common as c
from comparison import overlay,errors,root_only

COLORS=['#0072B2','#D55E00','#009E73']; SPLIT_COLORS={'train':'#0072B2','val':'#E69F00','test':'#009E73'}
KEYS=['Precision','Recall','Dice','IoU']

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--split-only',action='store_true');a=p.parse_args()
    out=a.run/'figures';out.mkdir(parents=True,exist_ok=True);tables=a.run/'tables'
    summary=c.read(tables/'summary.json') if not a.split_only else {};agg=summary.get('aggregate',[]);tiles=c.csvread(a.run/'data/tile_allocation.csv')
    tiles=[r for r in tiles if not r['tile_id'].endswith('_S12')]
    periods=c.csvread(tables/'per_period_metrics.csv') if not a.split_only else [];groups=c.csvread(tables/'four_date_pooled_tile_metrics.csv') if not a.split_only else []
    lengths=c.csvread(tables/'root_length_pairs.csv') if not a.split_only else [];data=c.Data(a.run);caps={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
        'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','legend.fontsize':7})
    def save(fig,name,caption,raster=False):
        captions={
            'Fig1_random_tile_split':'The fixed random25/5/6base-position allocation is retained. Two overlapping native256px views per training base position yield200training views; original20validation/24test coordinates remain. Added windows exclude all held-out pixels. All four dates at a window share one split. Original validation labels select weights/thresholds; final presentation uses revised references. These are correlated views, not200independent acquisitions.',
            'Fig2_test_segmentation_metrics':'Final-model agreement with REVISED model-assisted references, pooled over24internal-test tile-date images. White circles show four date-specific scores, not independent replicates or confidence intervals. All3models use the same6test coordinates. Weights/thresholds were frozen using ORIGINAL validation labels. Revised references are not independent manual truth and cannot establish external-tube accuracy. Original-reference metrics are retained separately as an audit CSV.',
            'Fig3_four_date_qualitative_comparison':'Four dates of the first test tile in sorted IDorder, not selected by model performance. Columns:originalRGB,revised reference,finalU-Net,finalDeepLabV3+,finalSegFormer-B0. Native256px coordinates and soil/droplet background are retained. Green denotes all binary roots, not new/persistent/extension classes. No display dilation or artificial connections. All6test tiles are supplied.',
            'Fig4_three_model_root_length_agreement':'Three final models,24correlated tile-date pairs. Reference lengths are skeletons derived from model-assisted revised masks, not independent manual root-length measurements. Native pixel units. Dashed line:identity; solid:OLS. IdentityR2(1-SSE/SST),OLSR2,RMSEandCCC are separately reported. Agreement can favor the model used to help revise references and does not prove independent anatomical accuracy.',
            'FigS3_Bland_Altman':'Descriptive Bland-Altman analysis against revised-reference skeleton lengths on24correlated tile-date pairs. Gray:mean difference; dashed:bias+/-1.96SD. These are observed reference-relative differences, not independent biological population limits or confidence intervals.',
            'FigS5_RGB_error_overlays':'S5errors are relative to REVISED references:greenTP(rootintersection),pinkFP(predicted root outside reference),orangeFN(reference root missed). Same fixed test tile/RGB asFig3; four rows are dates and three columns are final models. No tolerance dilation or reconnection. These are reference-relative categories, not independently verified biological correctness.',
            'Fig5_background_removed_local_RGB':'Same four dates and fixed test tile asFig3. Columns:revised reference,finalU-Net,finalDeepLabV3+,finalSegFormer-B0. SourceRGB is retained within each root mask and background set white. No display dilation, smoothing or enhancement. Full-upper and all6test-tile comparisons are provided; full fields include fitting pixels and are qualitative.',
            'FigS6_background_removed_local_masks':'Same comparison asFig5, using black roots on white background. Actual masks are displayed without dilation or artificial connections. References are revised, model-assisted masks, not independent manual truth.'}
        neutral={
            'Fig1_random_tile_split':'A fixed random allocation assigns25/5/6base positions to training/validation/internal test. Each training position contributes two overlapping native256px views; the four dates always share one split. Counts:200/20/24views. Added training windows exclude all held-out pixels. Views are correlated and do not represent independent biological acquisitions.',
            'Fig2_test_segmentation_metrics':'Precision,Recall,Dice andIoU against reference annotations on24internal-test tile-date views. Counts are pooled over all valid pixels. White circles show date-specific scores. Model weights and thresholds were frozen using the selection-validation set; test coordinates were excluded from training. Reference construction and scope are defined inMethods.',
            'Fig3_four_date_qualitative_comparison':'Four dates of the first test tile in sorted IDorder. Columns:originalRGB,reference annotation,U-Net,DeepLabV3+,SegFormer-B0. Native256px coordinates and soil/droplet backgrounds are retained. Green denotes root foreground. No display dilation or artificial connections. All6test tiles are provided.',
            'Fig4_three_model_root_length_agreement':'Three final models on24correlated tile-date pairs. Reference and predicted lengths are derived from mask skeletons in native pixels. Dashed line:identity; solid:OLS. IdentityR2,OLSR2,RMSE andCCC are reported separately. The reference and measurement procedure are defined inMethods.',
            'FigS1_learning_curves':'Training loss and actual selection-validation Dice at threshold0.5during the20epoch fine-tuning run. All models used64augmented draws per epoch,batch2. Dotted lines show the selected epochs. Thresholds were subsequently fixed using selection-validation data. Curves are single runs, not biological uncertainty estimates.',
            'FigS2_period_specific_Dice':'Date-specific Dice against reference annotations on validation/test coordinates. Common0-1color scale. Model and threshold selection precede this report; reference definitions are inMethods. Dates are repeated observations of one location.',
            'FigS3_Bland_Altman':'Length deviation analysis on24correlated tile-date pairs. Gray line:mean predicted-minus-reference length. Dashed lines:bias+/-1.96SD. Units are native pixels; reference lengths are derived from mask skeletons. These are descriptive observed limits, not independent population confidence intervals.',
            'FigS4_paired_test_tile_Dice':'Each line connects the same test coordinate across the three final models. Each tile score pools valid pixel counts from its four dates. All6test coordinates are retained. Adjacent tiles and dates may be correlated; no independent-replicate significance test is attached.',
            'FigS5_RGB_error_overlays':'Same fixed test tile and RGB backgrounds asFig3. Relative to reference annotations:greenTP(root intersection),pinkFP(predicted root outside reference),orangeFN(reference root missed). Rows:four dates; columns:three final models. No tolerance dilation or root reconnection is applied.',
            'Fig5_background_removed_local_RGB':'Four dates of the same fixed test tile. Columns:reference annotation,U-Net,DeepLabV3+,SegFormer-B0. SourceRGB is retained on root pixels and background set white. No mask dilation,smoothing or intensity enhancement. Full-upper and all6test-tile comparisons are supplied; full fields include fitting pixels and are qualitative.',
            'FigS6_background_removed_local_masks':'The same four-date comparison asFig5, rendered as black root foreground on white. Actual masks are displayed without dilation or artificial connections.'}
        caption=neutral.get(name,captions.get(name,caption))
        caption=re.sub(r'(?<=[a-z])(?=\d)',' ',caption)
        caption=re.sub(r'(?<=\d)(?=px\b|dates\b|epochs\b|positions\b|SD\b)',' ',caption)
        caption=re.sub(r',\s*',', ',caption)
        fig.savefig(out/(name+'.png'),dpi=600,bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
        if not raster:fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        with Image.open(out/(name+'.png')) as im:im.save(out/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
        plt.close(fig);caps[name]=caption;print('FIGURE',name,flush=True)
    def pooled(model,split):return next(r for r in agg if r['model']==model and r['split']==split)

    fig=plt.figure(figsize=(7.2,4.5),layout='constrained');gs=fig.add_gridspec(2,1,height_ratios=[3.5,1.2])
    ax=fig.add_subplot(gs[0]);rgb=np.array(Image.open(a.run/'data/upper/T1_RGB.png'));ax.imshow(rgb,interpolation='nearest')
    for r in tiles:
        x,y=int(r['x']),int(r['y']);co=SPLIT_COLORS[r['split']]
        ax.add_patch(Rectangle((x,y),256,256,fill=False,color=co,lw=1.2))
        ax.text(x+10,y+12,r['tile_id'],va='top',color=co,fontsize=6,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1.3))
    ax.set_axis_off();ax.set_title('(a) Direct native tiling followed by one fixed random allocation',loc='left')
    ax=fig.add_subplot(gs[1]);ax.set_axis_off()
    for i,(split,label,n) in enumerate([('train','Training',25),('val','Validation',5),('test','Internal test',6)]):
        x=i/3;ax.add_patch(Rectangle((x,.18),.025,.65,color=SPLIT_COLORS[split],transform=ax.transAxes))
        ax.text(x+.04,.72,f'{label}: {n} tile positions',transform=ax.transAxes,fontsize=8)
        ax.text(x+.04,.38,'25 base tiles × 2 views × 4 dates = 200' if split=='train' else f'{n} × 4 dates = {n*4} RGB patches',transform=ax.transAxes,fontsize=7)
    fig.suptitle('Four dates at the same tile position always share one split',fontsize=10)
    save(fig,'Fig1_random_tile_split','Native upper field,2541×1108px. The previous fixed random allocation of25/5/6base tile positions is unchanged. Training has two native256px views per base position, the second translated12px in x/y inward at image boundaries, yielding200training views; original20validation/24test images are retained. Added training windows explicitly exclude all held-out pixels; within-train views overlap and are not independent acquisitions. Only training targets are revised with U-Net and RGB tubular support. All three previous matching models are fine-tuned for20epochs, with epoch0eligible for retention. Original validation and test targets remain the selection/primary evaluation references. No easy-tile selection or test target edits.',True)
    if a.split_only:
        c.js(out/'captions.json',caps);return

    fig,axs=plt.subplots(1,4,figsize=(7.2,2.7),layout='constrained')
    for ax,key in zip(axs,KEYS):
        for j,model in enumerate(c.MODELS):
            score=pooled(model,'test')[key];dots=[float(r[key]) for r in periods if r['model']==model and r['split']=='test']
            ax.barh(j,score,height=.42,color=COLORS[j],alpha=.85)
            ax.scatter(dots,j+np.array([-.14,-.045,.045,.14]),s=13,facecolors='white',edgecolors='#333333',lw=.6,zorder=4)
            ax.text(.98,j-.29,f'{score:.3f}',ha='right',fontsize=7.5)
        ax.set(xlim=(0,1),ylim=(2.6,-.65),xlabel=key,xticks=[0,.5,1]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_yticks(range(3),[c.NAMES[m] for m in c.MODELS] if key=='Precision' else ['']*3)
    fig.suptitle('Final models · reference agreement · internal test',fontsize=10)
    fig.legend(handles=[Line2D([],[],marker='o',mfc='white',mec='#333333',ls='',label='Individual dates; descriptive repeated observations')],loc='outside lower center',frameon=False)
    save(fig,'Fig2_test_segmentation_metrics','Primary pilot scores pool TP,FP,FN over all24 held-out tile-date images, counting each valid pixel once. White circles show the four date-specific scores, not independent replicates or confidence intervals. The same6 test tile positions and unchanged human masks were used for all three models. Checkpoints and thresholds were frozen using validation only; no test-based postprocessing was performed. This is internal same-position testing, not independent-tube evaluation.')

    chosen=sorted(r['tile_id'] for r in tiles if r['split']=='test')[0]
    fig,axs=plt.subplots(4,5,figsize=(7.2,6.3),layout='constrained')
    for i,t in enumerate(c.PERIODS):
        r=next(r for r in data.rows if r['tile_id']==chosen and r['period']==t);rgb,gt=data.load(r)
        axs[i,0].imshow(rgb,interpolation='nearest');axs[i,1].imshow(overlay(rgb,gt==255),interpolation='nearest')
        for j,model in enumerate(c.MODELS):
            mask=np.array(Image.open(a.run/'native_predictions'/model/'test'/(r['image_id']+'_prediction.png')))>0
            axs[i,j+2].imshow(overlay(rgb,mask),interpolation='nearest')
        for j,ax in enumerate(axs[i]):
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title((['Original RGB','Reference annotation']+[c.NAMES[m] for m in c.MODELS])[j])
            if j==0:ax.set_ylabel(t,rotation=0,labelpad=14,va='center')
    fig.suptitle(f'Fixed held-out tile {chosen} · same native coordinates at four dates',fontsize=10)
    save(fig,'Fig3_four_date_qualitative_comparison','Four dates of the first test tile in sorted tile-ID order(predefined by the allocation, not selected for model performance). Columns show original RGB, human annotation overlay, U-Net, DeepLabV3+ and SegFormer-B0 overlays. Every column retains the original soil/droplet RGB background. The same green root color(0,158,115;alpha0.75) is applied only to each actual foreground mask, with no mask dilation or artificial root connection. Displayed regions are256×256 native pixels at identical coordinates across dates/models. All6 test tiles are also delivered with native individual PNGs.',True)

    fig,axs=plt.subplots(1,3,figsize=(7.2,2.9),layout='constrained');markers=['o','s','^','D']
    lim=max(max(float(r['reference_mask_skeleton_length_px']),float(r['predicted_skeleton_length_px'])) for r in lengths)*1.06
    for j,model in enumerate(c.MODELS):
        ax=axs[j];rec=next(r for r in summary['root_length'] if r['model']==model)
        for t,marker in zip(c.PERIODS,markers):
            rr=[r for r in lengths if r['model']==model and r['period']==t]
            ax.scatter([float(r['reference_mask_skeleton_length_px']) for r in rr],[float(r['predicted_skeleton_length_px']) for r in rr],s=17,marker=marker,color=COLORS[j],alpha=.7)
        ax.plot([0,lim],[0,lim],'--',color='#777777',lw=.8)
        xx=np.array([0,lim]);ax.plot(xx,rec['slope']*xx+rec['intercept_px'],color=COLORS[j],lw=1)
        ax.text(.04,.96,f'Identity R² = {rec["R2_identity"]:.3f}\nOLS R² = {rec["R2_OLS"]:.3f}\nRMSE = {rec["RMSE_px"]:.1f} px\nCCC = {rec["CCC"]:.3f}',transform=ax.transAxes,va='top',fontsize=7.2)
        ax.set(xlim=(0,lim),ylim=(0,lim),aspect='equal',title=c.NAMES[model],xlabel='Reference length (px)',ylabel='Predicted length (px)' if j==0 else '')
    fig.legend(handles=[Line2D([],[],marker=m,color='#444444',ls='',label=t) for t,m in zip(c.PERIODS,markers)],loc='outside lower center',ncol=4,frameon=False)
    save(fig,'Fig4_three_model_root_length_agreement','Three-model length agreement on the same24 correlated tile-date pairs(6 tile positions repeated at4 dates). Human references are skeleton lengths derived from the supplied masks, not separate manual root-length measurements. Native-pixel units are used because no physical scale was supplied. Dashed line is identity; solid line is OLS. Displayed R² is1−SSE/SST relative to identity, not squared correlation or OLS R²; both definitions are available in the data table. High length agreement alone does not demonstrate accurate root boundaries or temporal identities.')

    fig,axs=plt.subplots(2,3,figsize=(7.2,4.5),layout='constrained')
    for j,model in enumerate(c.MODELS):
        hist=c.csvread(a.run/model/'history.csv');epochs=[int(r['epoch']) for r in hist]
        axs[0,j].plot(epochs,[float(r['train_loss']) for r in hist],color=COLORS[j],lw=1)
        rr=[r for r in hist if r['validation_Dice']];axs[1,j].plot([int(r['epoch']) for r in rr],[float(r['validation_Dice']) for r in rr],color=COLORS[j],lw=1,marker='o',ms=2)
        selected=pooled(model,'test')['selected_epoch'];axs[1,j].axvline(selected,ls=':',color='#777777',lw=.8)
        axs[0,j].set_title(c.NAMES[model]);axs[0,j].set_ylabel('Training loss' if j==0 else '');axs[1,j].set(ylim=(0,1),xlabel='Epoch',ylabel='Selection-val Dice at 0.5' if j==0 else '')
        for ax in axs[:,j]:ax.grid(alpha=.15);ax.set_xlim(0,20)
    save(fig,'FigS1_learning_curves','All three matching previous checkpoints were fine-tuned using revised training masks, the identical random tile allocation,64 augmented draws per epoch and20 epochs. Epoch0 is eligible for retention. ORIGINAL validation Dice at threshold0.5 selected each checkpoint; the threshold was subsequently calibrated using ORIGINAL validation only. Test labels were never used for selection. Dotted lines mark selected epochs. Curves are single runs, not uncertainty estimates.')

    fig,axs=plt.subplots(1,2,figsize=(7.2,3),layout='constrained')
    for ax,split in zip(axs,['val','test']):
        mat=np.array([[next(float(r['Dice']) for r in periods if (r['model'],r['split'],r['period'])==(model,split,t)) for t in c.PERIODS] for model in c.MODELS])
        im=ax.imshow(mat,vmin=0,vmax=1,cmap='cividis',aspect='auto')
        for y in range(3):
            for x in range(4):ax.text(x,y,f'{mat[y,x]:.3f}',ha='center',va='center',color='white' if mat[y,x]<.55 else '#111111')
        ax.set_xticks(range(4),c.PERIODS);ax.set_yticks(range(3),[c.NAMES[m] for m in c.MODELS] if split=='val' else ['']*3)
        ax.set_title('Validation coordinates · reference' if split=='val' else 'Internal test · reference')
    fig.colorbar(im,ax=axs,fraction=.025,pad=.025,label='Dice',ticks=[0,.5,1])
    save(fig,'FigS2_period_specific_Dice','Date-specific Dice against revised references on validation/test coordinates, fixed0-1color scale. Model/threshold selection actually used ORIGINAL validation references; these plotted revised-reference values were computed afterward and were not used for selection. Dates are correlated repeated observations, not independent biological replicates.')

    fig,axs=plt.subplots(1,3,figsize=(7.2,2.7),layout='constrained');err=[]
    for j,model in enumerate(c.MODELS):
        rr=[r for r in lengths if r['model']==model];x=np.array([float(r['reference_mask_skeleton_length_px']) for r in rr]);y=np.array([float(r['predicted_skeleton_length_px']) for r in rr]);rec=next(r for r in summary['root_length'] if r['model']==model)
        axs[j].scatter((x+y)/2,y-x,s=14,color=COLORS[j],alpha=.7)
        for key,ls in [('bias_px','-'),('LoA_lower_px','--'),('LoA_upper_px','--')]:axs[j].axhline(rec[key],ls=ls,color='#777777',lw=.8)
        axs[j].axhline(0,color='#BBBBBB',lw=.5);axs[j].set(title=c.NAMES[model],xlabel='Mean length (px)',ylabel='Prediction − reference (px)' if j==0 else '')
        err.extend([*(y-x),rec['LoA_lower_px'],rec['LoA_upper_px']])
    mx=max(abs(np.asarray(err)))*1.1
    for ax in axs:ax.set_ylim(-mx,mx)
    save(fig,'FigS3_Bland_Altman','Descriptive Bland–Altman plots for the same24 correlated tile-date pairs. Solid gray line is mean difference; dashed lines are bias±1.96SD. These describe observed agreement, not biological population limits or independent-sample confidence intervals. Human reference lengths are mask-derived skeleton lengths in native pixels.')

    fig,ax=plt.subplots(figsize=(7.2,3),layout='constrained');testtiles=sorted({r['tile_id'] for r in groups if r['split']=='test'})
    for i,tile in enumerate(testtiles):
        vals=[next(float(r['Dice']) for r in groups if (r['model'],r['split'],r['tile_id'])==(m,'test',tile)) for m in c.MODELS]
        ax.plot(range(3),vals,color='#CCCCCC',lw=.8,zorder=1)
        for j,v in enumerate(vals):ax.scatter(j,v,s=20,color=COLORS[j],zorder=2)
        ax.text(2.04,vals[2],tile,fontsize=6.5,va='center')
    ax.set(xlim=(-.3,2.5),ylim=(0,1),xticks=range(3),xticklabels=[c.NAMES[m] for m in c.MODELS],ylabel='Four-date pooled tile Dice')
    ax.grid(axis='y',alpha=.15);save(fig,'FigS4_paired_test_tile_Dice','Each gray line links the same test tile position across three models, with TP/FP/FN pooled across its4 dates. All6 randomly assigned test positions are retained. This plot exposes heterogeneity rather than hiding difficult tiles in a mean. Adjacent tiles may contain portions of the same root; lines are not independent biological replicates and no significance test is attached.')

    # All test tiles get separate comparable four-date panels in the result package.
    for tile in sorted(r['tile_id'] for r in tiles if r['split']=='test'):
        fig,axs=plt.subplots(4,5,figsize=(7.2,6.3),layout='constrained')
        for i,t in enumerate(c.PERIODS):
            r=next(r for r in data.rows if r['tile_id']==tile and r['period']==t);rgb,gt=data.load(r)
            ims=[rgb,overlay(rgb,gt==255)]+[overlay(rgb,np.array(Image.open(a.run/'native_predictions'/m/'test'/(r['image_id']+'_prediction.png')))>0) for m in c.MODELS]
            for j,(ax,im) in enumerate(zip(axs[i],ims)):
                ax.imshow(im,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                if i==0:ax.set_title((['RGB','Reference annotation']+[c.NAMES[m] for m in c.MODELS])[j])
                if j==0:ax.set_ylabel(t,rotation=0,labelpad=12)
        fig.suptitle('Held-out '+tile,fontsize=10);fig.savefig(out/('All_test_'+tile+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(4,3,figsize=(7.2,7.6),layout='constrained')
    for i,t in enumerate(c.PERIODS):
        r=next(r for r in data.rows if r['tile_id']==chosen and r['period']==t);rgb,gt=data.load(r)
        for j,model in enumerate(c.MODELS):
            pred=np.array(Image.open(a.run/'native_predictions'/model/'test'/(r['image_id']+'_prediction.png')))>0
            axs[i,j].imshow(errors(rgb,gt,pred),interpolation='nearest');axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
            if i==0:axs[i,j].set_title(c.NAMES[model])
            if j==0:axs[i,j].set_ylabel(t,rotation=0,labelpad=12)
    fig.legend(handles=[Patch(color=co,label=label) for co,label in [('#009E73','Correct root'),('#CC79A7','False positive'),('#E69F00','Missed root')]],loc='outside lower center',ncol=3,frameon=False)
    save(fig,'FigS5_RGB_error_overlays','Same fixed test tile and native RGB backgrounds as Fig3, with green true-positive root pixels,pink false-positive pixels and orange missed human-labelled root pixels. No tolerance dilation or root reconnection is applied. Error regions are defined strictly relative to unchanged supplied masks, which may not perfectly delineate anatomical widths. Each row is one date; each column one model.',True)
    for style,name in [('RGB','Fig5_background_removed_local_RGB'),('black_root','FigS6_background_removed_local_masks')]:
        fig,axs=plt.subplots(4,4,figsize=(7.2,7.2),layout='constrained')
        for i,t in enumerate(c.PERIODS):
            r=next(r for r in data.rows if r['tile_id']==chosen and r['period']==t);rgb,gt=data.load(r)
            masks=[gt==255]+[np.array(Image.open(a.run/'native_predictions'/m/'test'/(r['image_id']+'_prediction.png')))>0 for m in c.MODELS]
            for j,mask in enumerate(masks):
                im=root_only(rgb,mask) if style=='RGB' else ~mask
                axs[i,j].imshow(im,cmap='gray',vmin=0,vmax=1,interpolation='nearest');axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
                if i==0:axs[i,j].set_title((['Reference annotation']+[c.NAMES[m] for m in c.MODELS])[j])
                if j==0:axs[i,j].set_ylabel(t,rotation=0,labelpad=12)
        fig.suptitle('Background removed · '+('original root RGB retained' if style=='RGB' else 'black roots on white'),fontsize=10)
        save(fig,name,'Four dates of the same fixed held-out tile as Fig3. Columns are human annotation,U-Net,DeepLabV3+ and SegFormer-B0. Background pixels are set to white; '+('actual RGB values at foreground pixels are retained.' if style=='RGB' else 'foreground masks are displayed as black roots on white.')+' No dilation, smoothing, reconnection or intensity enhancement is applied. Full native upper-field comparisons and all6 test tile positions are delivered separately; whole-field views include fitting pixels and are qualitative.',True)
    c.js(out/'captions.json',caps)
    (out/'Figure_legends_EN.txt').write_text('\n\n'.join(name+'\n'+text for name,text in caps.items())+'\n')

if __name__=='__main__':main()
