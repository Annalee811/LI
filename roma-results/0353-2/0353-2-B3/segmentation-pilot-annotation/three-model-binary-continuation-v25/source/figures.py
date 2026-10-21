"""Replay and conservative continuation diagnostics; all measured epochs retained."""
import argparse,csv,hashlib,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image,ImageCms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,ScalarFormatter

MODELS=['unet','deeplab','segformer'];NAMES=['U-Net','DeepLabV3+','SegFormer-B0'];COLORS=['#0072B2','#D55E00','#009E73']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def js(p,obj):Path(p).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def csvout(p,rr):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def counts(gt,pred):
    valid=gt!=128;g=gt==255;p=pred.astype(bool)
    return np.array([(g&p&valid).sum(),(~g&p&valid).sum(),(g&~p&valid).sum(),(~g&~p&valid).sum()],np.int64)
def score(cc):
    tp,fp,fn,tn=map(int,cc);return dict(TP=tp,FP=fp,FN=fn,TN=tn,Precision=tp/(tp+fp),Recall=tp/(tp+fn),Dice=2*tp/(2*tp+fp+fn),IoU=tp/(tp+fp+fn),Accuracy=(tp+tn)/(tp+fp+fn+tn),Specificity=tn/(tn+fp),Balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)))
def roc_pr(y,p):
    order=np.argsort(-p,kind='stable');yp=y[order].astype(np.int64);sp=p[order]
    ix=np.r_[np.flatnonzero(sp[:-1]!=sp[1:]),len(sp)-1];tp=np.cumsum(yp)[ix];fp=ix+1-tp;positive=int(y.sum());negative=len(y)-positive
    fpr=np.r_[0.,fp/negative];tpr=np.r_[0.,tp/positive];precision=tp/(tp+fp);recall=tp/positive
    area=float(np.trapezoid(tpr,fpr));ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    assert fpr[-1]==1 and tpr[-1]==1 and np.all(np.diff(fpr)>=0) and np.all(np.diff(tpr)>=0)
    # All thresholds/ties contribute to AUC/AP; this subset is for rendering only.
    ir=np.unique(np.r_[0,np.linspace(0,len(fpr)-1,min(5000,len(fpr))).astype(int),len(fpr)-1])
    ip=np.unique(np.r_[0,np.linspace(0,len(recall)-1,min(5000,len(recall))).astype(int),len(recall)-1])
    return dict(fpr=fpr[ir],tpr=tpr[ir],recall=np.r_[0.,recall[ip]],precision=np.r_[1.,precision[ip]],AUC=area,AP=ap,unique_score_thresholds=len(ix),root_prevalence=positive/len(y),valid_pixels=len(y),positive_pixels=positive)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);a=ap.parse_args();out=a.run/'report';out.mkdir(parents=True,exist_ok=True);view=a.source/'revised_report';figures=out/'figures';figures.mkdir(exist_ok=True)
    summary=read(view/'tables/summary.json')
    manifests={ref:rows(folder/'data/manifest.csv') for ref,folder in [('selection',a.run),('final',view)]};original_val=[r for r in manifests['selection'] if r['split']=='val'];test=[r for r in manifests['final'] if r['split']=='test'];assert len(original_val)==20 and len(test)==24 and len({r['tile_id'] for r in test})==6
    npos=0;nvalid=0;inputs=[]
    for r in original_val:
        p=a.run/r['mask_path'];assert sha(p)==r['mask_sha256'];gt=np.array(Image.open(p));npos+=int((gt==255).sum());nvalid+=int((gt!=128).sum())
    print('VERSION_AND_FIXED_VALIDATION_COUNTS_VERIFIED',npos,nvalid,flush=True)
    curves={};history_table=[];schedule=[];selected={};configs={};replay_info={};primary_loss_rows=[]
    for m in MODELS:
        hp=a.run/m/'history.csv';cfg=read(a.run/m/'training_run.json');sel=read(a.run/m/'selection.json');hh=rows(hp);selected[m]=int(sel['selected_epoch']);configs[m]=cfg;replay_info[m]=read(a.run/m/'historical_replay_complete.json')
        assert len(hh)==cfg['epochs_completed'] and [int(r['epoch']) for r in hh]==list(range(1,len(hh)+1))
        assert all(r['validation_loss'] for r in hh),'A loss value is missing; never fill it with a proxy'
        curves[m]=dict(epoch=np.array([int(r['epoch']) for r in hh]),loss=np.array([float(r['train_loss']) for r in hh]),val_loss=np.array([float(r['validation_loss']) for r in hh]),metrics=[])
        for r in hh:
            epoch=int(r['epoch']);lr=float(r['learning_rate']);schedule.append(dict(model=m,epoch=epoch,learning_rate=lr,source=r['phase']))
            cc=np.array([int(float(r['validation_'+k])) for k in ['TP','FP','FN','TN']]);met=score(cc)
            for key in ['Dice','IoU','Accuracy','Precision','Recall']:assert abs(met[key]-float(r['validation_'+key]))<1e-12
            assert met['TP']+met['FN']==npos and sum(cc)==nvalid
            rec=dict(model=m,epoch=epoch,**met);curves[m]['metrics'].append(rec);history_table.append(rec)
        inputs.append(dict(kind='replay_and_continuation_history',model=m,path=str(hp.resolve()),history_SHA256=sha(hp),original_history_SHA256=sha(a.source/m/'history.csv')))
    csvout(out/'validation_epoch_metrics.csv',history_table);csvout(out/'learning_rate_schedule.csv',schedule);js(out/'historical_replay_comparison.json',replay_info)
    print('ALL_MEASURED_EPOCH_LOSSES_AND_COUNTS_VERIFIED',flush=True)
    evaluations=[];prob_curves={};prob_sources=[]
    for m in MODELS:
        sel=read(a.run/m/'selection.json');threshold=float(sel['threshold']);truth=[];probabilities=[];total=np.zeros(4,np.int64)
        for r in test:
            gp=view/r['mask_path'];assert sha(gp)==r['mask_sha256'];gt=np.array(Image.open(gp));pp=a.run/m/'probabilities/test'/(r['image_id']+'_probability.npy');pr=np.load(pp);assert pr.shape==gt.shape==(256,256) and np.isfinite(pr).all() and pr.min()>=0 and pr.max()<=1
            valid=gt!=128;truth.append((gt[valid]==255));probabilities.append(pr[valid]);total+=counts(gt,pr>=threshold);prob_sources.append(dict(model=m,image_id=r['image_id'],probability_path=str(pp.resolve()),probability_SHA256=sha(pp),reference_path=str(gp.resolve()),reference_SHA256=sha(gp)))
        met=score(total);published=next(r for r in summary['aggregate'] if r['model']==m and r['split']=='test')
        if configs[m]['baseline_retained']:
            for key in ['TP','FP','FN','TN','Precision','Recall','Dice','IoU']:assert met[key]==published[key],(m,key,met[key],published[key])
        curve=roc_pr(np.concatenate(truth),np.concatenate(probabilities));prob_curves[m]=curve;evaluations.append(dict(model=m,name=NAMES[MODELS.index(m)],threshold=threshold,**met,ROC_AUC=curve['AUC'],Average_precision=curve['AP'],root_prevalence=curve['root_prevalence'],valid_test_pixels=curve['valid_pixels'],test_RGB_patches=24,test_spatial_positions=6,selected_epoch=selected[m],old_final_Dice=published['Dice'],Dice_change=met['Dice']-published['Dice'],baseline_retained=configs[m]['baseline_retained']))
        csvout(out/(m+'_ROC_plot_points.csv'),[dict(FPR=float(x),TPR=float(y)) for x,y in zip(curve['fpr'],curve['tpr'])]);print('FINAL_TEST_COUNTS_AND_ROC_VERIFIED',m,'Dice',met['Dice'],'AUC',curve['AUC'],'AP',curve['AP'],flush=True)
    csvout(out/'final_test_metrics.csv',evaluations)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#E5E9ED','grid.linewidth':.5,'pdf.fonttype':42,'savefig.facecolor':'white'})
    sRGB=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes();outputs=[];captions={}
    def save(fig,name,caption):
        fig.canvas.draw();png=out/(name+'_preview.png');fig.savefig(png,dpi=180);temp=out/(name+'_600.png');fig.savefig(temp,dpi=600);plt.close(fig)
        p=figures/(name+'.tif')
        with Image.open(temp) as im:im.convert('RGB').save(p,compression='tiff_lzw',dpi=(600,600),icc_profile=sRGB)
        with Image.open(p) as im:im.load();assert im.mode=='RGB' and im.n_frames==1 and tuple(im.info['dpi'])==(600.,600.);size=list(im.size)
        temp.unlink();outputs.append(dict(file=p.name,bytes=p.stat().st_size,SHA256=sha(p),size=size,DPI=600));captions[name]=caption;print('SAVED',p.name,flush=True)
    max_epoch=max(int(q['epoch'][-1]) for q in curves.values())
    def decorate(ax,last=max_epoch):
        ax.grid(alpha=.6);ax.set_axisbelow(True);ax.tick_params(length=3,width=.65);ax.set_xlim(1,last+.5);ax.xaxis.set_major_locator(MultipleLocator(5 if last<=40 else 10));ax.axvline(20.5,color='#9BA1A8',lw=.8,ls=':')
    global_loss_max=max(max(q['val_loss'].max(),q['loss'].max()) for q in curves.values())*1.12
    for m,name,color in zip(MODELS,NAMES,COLORS):
        curve=curves[m];last=int(curve['epoch'][-1]);fig,ax=plt.subplots(figsize=(3.65,3.0));fig.subplots_adjust(left=.18,right=.97,top=.86,bottom=.26)
        near=replay_info[m]['near_original_training_loss'] and replay_info[m]['near_original_logged_Dice'];display_train=curve['loss'].copy();display_val=curve['val_loss'].copy()
        if not near:
            # A failed replay is never used to fill the missing old validation
            # log. Preserve the original training history and the honest gap.
            original=rows(a.source/m/'history.csv');display_train[:20]=[float(r['train_loss']) for r in original];display_val[:20]=np.nan
        for i,epoch in enumerate(curve['epoch']):primary_loss_rows.append(dict(model=m,epoch=int(epoch),training_loss=float(display_train[i]),validation_loss=float(display_val[i]) if np.isfinite(display_val[i]) else None,source=('close_measured_replay' if near else 'original_train_only_validation_unrecorded') if epoch<=20 else 'measured_final_weight_continuation'))
        ax.axvspan(1,20.5,color='#F2F4F6',zorder=0);ax.plot(curve['epoch'],display_train,color='#0072B2',lw=1.2,label='Training Loss');ax.plot(curve['epoch'],display_val,color='#D55E00',lw=1.2,label='Validation Loss');decorate(ax,last);ax.set(title=name+(' · all measured epochs' if near else ' · recorded loss history'),xlabel='Epoch',ylabel='Loss',ylim=(0,global_loss_max));ax.legend(frameon=False,fontsize=7,loc='upper right');fig.text(.56,.075,'1–20: replay measurements' if near else '1–20: original training loss; validation unrecorded',ha='center',fontsize=7 if near else 6,color='#5F6873');fig.text(.56,.03,'21 onward: final-weight continuation',ha='center',fontsize=7,color='#5F6873')
        phase_caption=('Epochs1–20 are actual near-replay measurements from archived starting weights, not originally recorded validation-loss values.' if near else 'The replay failed the advance proximity check. Epochs1–20 therefore show the archived original training loss only; original validation loss was unrecorded and remains visibly missing. The separate replay audit figure displays every replay measurement without substituting it for the original log.')
        save(fig,'Fig01_'+m+'_training_and_validation_loss','Complete recorded range for '+name+'. '+phase_caption+' Both plotted data losses use the same objective. Original histories remain untouched and deviations are tabulated. At the dotted boundary, the existing final selected weights are reloaded and continuation begins at index21 with a new optimizer. Validation labels retain the original selection-validation reference. All completed continuation epochs are displayed without smoothing or selected-epoch truncation.')
        if not near:
            fig,ax=plt.subplots(figsize=(3.65,3.0));fig.subplots_adjust(left=.18,right=.97,top=.86,bottom=.25);ax.plot(curve['epoch'],curve['loss'],color='#0072B2',lw=1.2,label='Measured Training Loss');ax.plot(curve['epoch'],curve['val_loss'],color='#D55E00',lw=1.2,label='Measured Validation Loss');ax.plot(range(1,21),display_train[:20],color='#8C959F',lw=1,ls='--',label='Archived Training Loss');decorate(ax,last);ax.set(ylim=(0,global_loss_max),xlabel='Monitoring epoch',ylabel='Loss',title=name+' · replay + continuation');ax.legend(frameon=False,fontsize=6.5,loc='upper right');fig.text(.56,.065,'Replay differs from archived history',ha='center',fontsize=7,color='#5F6873');fig.text(.56,.02,'Not a reconstruction of missing original validation loss',ha='center',fontsize=6,color='#5F6873')
            save(fig,'FigS01_'+m+'_separate_measured_replay','All20new replay train/validation measurements and every additional continuation epoch for '+name+', compared with the untouched original20epoch training-loss records. The dotted line marks reloading the original selected checkpoint and starting a new optimizer at index21. This replay failed at least one advance proximity criterion (maximum relative training-loss difference≤1% and maximum logged Dice difference≤0.005). It is an independent replay audit and is not used to fill the missing original validation-loss record. All points, including unfavorable deviations, are retained.')
        rr=[r for r in schedule if r['model']==m];fig,ax=plt.subplots(figsize=(3.65,2.85));fig.subplots_adjust(left=.20,right=.97,top=.85,bottom=.24);ax.plot([r['epoch'] for r in rr],[r['learning_rate'] for r in rr],color=color,lw=1.3);decorate(ax,last);ax.set(title=name+' · learning rate',xlabel='Epoch',ylabel='Learning rate',ylim=(0,9e-5));ax.ticklabel_format(axis='y',style='sci',scilimits=(-5,-5),useMathText=True);fig.text(.56,.04,'Cosine replay → validation-loss plateau decay',ha='center',fontsize=6.8,color='#5F6873')
        save(fig,'Fig02_'+m+'_learning_rate','Actual rates used in replay and continuation: original20epoch cosine formula, then1e-5warm restart and factor0.5plateau adaptation with a1e-7floor. Dotted line is the stage boundary. The complete measured epoch range is shown.')
    csvout(out/'primary_loss_plot_data.csv',primary_loss_rows)
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.1));fig.subplots_adjust(left=.085,right=.985,top=.89,bottom=.17,wspace=.24,hspace=.43)
    for k,(ax,key,label,lim) in enumerate(zip(axs.ravel(),['Dice','IoU','Accuracy','Balanced_accuracy'],['Dice','IoU','Pixel accuracy','Balanced accuracy'],[(.25,.80),(.1,.7),(.9,1.0),(.5,.95)])):
        for m,name,color in zip(MODELS,NAMES,COLORS):
            rr=curves[m]['metrics'];xx=[r['epoch'] for r in rr];yy=[r[key] for r in rr];ax.plot(xx,yy,color=color,lw=1.15,marker='o',ms=2.6,label=name)
            # The retained old checkpoint is not the replay checkpoint at the
            # same epoch. Mark only a winner measured during continuation.
            if not configs[m]['baseline_retained']:
                q=next(r for r in rr if r['epoch']==selected[m]);ax.scatter([selected[m]],[q[key]],s=24,marker='D',facecolor='white',edgecolor=color,lw=1,zorder=5)
            assert all(lim[0]<=v<=lim[1] for v in yy),(key,min(yy),max(yy))
        decorate(ax);ax.set(xlabel='Epoch',ylabel=label,ylim=lim);ax.text(-.12,1.05,chr(97+k),transform=ax.transAxes,fontweight='bold',fontsize=10)
    fig.suptitle('Replay + continuation · validation metrics at threshold 0.5',fontsize=10,fontweight='semibold',y=.99);fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.035),ncol=3,frameon=False,fontsize=7.5)
    save(fig,'Fig03_validation_metrics_vs_epoch','Every completed replay and continuation epoch is evaluated on the unchanged original selection-validation set at threshold0.5. Dice,IoU,pixel accuracy and balanced accuracy are computed from saved integer counts and verified against recorded values. Epochs1–20are new replay measurements, not missing original logs presented as original. The dotted boundary marks reloading the existing final checkpoint. Open diamonds, if present, mark newly selected continuation checkpoints; retained historical checkpoints are not marked as replay checkpoints. No test data select weights or stop training.')
    fig,ax=plt.subplots(figsize=(7.2,3.2));fig.subplots_adjust(left=.085,right=.985,top=.85,bottom=.25);keys=['Precision','Recall','Dice','IoU','Accuracy'];x=np.arange(len(keys));width=.23
    for j,(row,name,color) in enumerate(zip(evaluations,NAMES,COLORS)):
        vv=[row[k] for k in keys];xx=x+(j-1)*width;ax.bar(xx,vv,width*.91,color=color,label=name)
        for xi,v in zip(xx,vv):ax.text(xi,v+.012,f'{v:.3f}',ha='center',va='bottom',fontsize=6.3)
    ax.set_xticks(x,['Precision','Recall','Dice','IoU','Pixel accuracy']);ax.set(ylim=(0,1.1),ylabel='Score',title='Final models · internal-test reference agreement');ax.grid(axis='y',alpha=.5);ax.set_axisbelow(True);fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.025),ncol=3,frameon=False,fontsize=7.5)
    save(fig,'Fig04_final_test_metric_comparison','Final test scores on the same 24 tile-date views at six spatial coordinates, using the revised reporting reference in the named final paper package. The same final reference and fixed thresholds are retained. Original baselines reproduce archived scores; any eligible new checkpoints are rescored from their own probabilities. Pixel accuracy is background-dominated; root-class Dice/IoU and Precision/Recall remain necessary. No external independent test images are available.')
    fig,axs=plt.subplots(1,3,figsize=(7.2,3.2));fig.subplots_adjust(left=.17,right=.85,top=.79,bottom=.24,wspace=.30)
    for j,(ax,row,name) in enumerate(zip(axs,evaluations,NAMES)):
        cm=np.array([[row['TN'],row['FP']],[row['FN'],row['TP']]]);norm=cm/cm.sum(1,keepdims=True);im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',aspect='equal')
        for y in range(2):
            for x in range(2):ax.text(x,y,f'{cm[y,x]:,}\n{norm[y,x]:.1%}',ha='center',va='center',fontsize=7,color='white' if norm[y,x]>.6 else '#142F46',linespacing=1.7)
        ax.set_xticks([0,1],['Background','Root'],fontsize=7);ax.set_yticks([0,1],['Background','Root'] if j==0 else ['','']);ax.set_xlabel('Predicted label',labelpad=7);ax.set_title(name+f"\nthreshold {row['threshold']:.3f}",fontsize=8,pad=10)
        if j==0:ax.set_ylabel('Reference label',labelpad=7)
    cax=fig.add_axes([.89,.30,.013,.42]);fig.colorbar(im,cax=cax,label='Row proportion',ticks=[0,.5,1]);fig.suptitle('Final internal-test confusion matrices',fontsize=10,fontweight='semibold',y=.99)
    save(fig,'Fig05_final_test_confusion_matrices','Matrices pool all valid pixels from 24 internal-test tile-date views. Rows are final reporting-reference labels; columns are predictions using the original validation-fixed threshold for each selected model. Cell text gives absolute counts and row percentages; a common 0-1 row-normalized color scale prevents background prevalence from masking root recall. Retained original baselines exactly reproduce the old package; new eligible checkpoints have freshly computed counts on the unchanged reference. The reference is not an independently collected external test.')
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.55));fig.subplots_adjust(left=.08,right=.985,top=.82,bottom=.24,wspace=.25)
    for m,name,color in zip(MODELS,NAMES,COLORS):
        q=prob_curves[m];axs[0].plot(q['fpr'],q['tpr'],color=color,lw=1.3,label=f"{name}: AUC = {q['AUC']:.4f}");axs[1].plot(q['recall'],q['precision'],color=color,lw=1.3,label=f"{name}: AP = {q['AP']:.4f}")
    axs[0].plot([0,1],[0,1],ls='--',lw=.8,color='#A7ADB4');prevalence=prob_curves['unet']['root_prevalence'];axs[1].axhline(prevalence,ls='--',lw=.8,color='#A7ADB4')
    for ax in axs:ax.set_xlim(0,1);ax.set_ylim(0,1.025);ax.grid(alpha=.5);ax.legend(loc='lower left',fontsize=6.5,frameon=True,edgecolor='white',facecolor='white',framealpha=.94)
    axs[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='a  ROC curves');axs[1].set(xlabel='Recall',ylabel='Precision',title='b  Precision–recall curves');fig.suptitle('Final internal-test probability discrimination',fontsize=10,fontweight='semibold',y=.98);fig.text(.53,.035,f'24 tile-date views · 6 positions · root prevalence {prevalence:.2%}',ha='center',fontsize=7,color='#5F6873')
    save(fig,'Fig06_final_test_ROC_AUC_and_PR','ROC/AUC and precision-recall/AP are computed from the selected models saved continuous float32 test probabilities, not binary masks. All valid pixels of the same 24 final-reference tile-date views are included with no sampling or difficult-region exclusion. Tied probability scores are grouped; AUC uses the full empirical ROC with trapezoidal integration and AP uses non-interpolated step integration. Plot traces retain up to 5000 display points, while AUC/AP include every unique threshold. PR is included because root/background pixels are imbalanced. Dates/tiles are correlated; no pixel-independence confidence intervals or external generalization claims are made.')
    js(out/'captions.json',captions);js(out/'source_probability_manifest.json',prob_sources)
    protocol=dict(source_final_package='0353-2-B3_final_paper_figures.zip',source_run=str(a.source.resolve()),training_run=str(a.run.resolve()),report_generator_performs_training=False,experiment_includes_continuation=True,annotations_modified=False,models=MODELS,epochs_each={m:configs[m]['epochs_completed'] for m in MODELS},selected_epochs=selected,available_figures=len(outputs),replay_measurements_explicitly_identified=True,close_replays_only_used_in_primary_loss_figures=True,unmatched_replay_validation_not_imputed_into_original_history=True,loss_plot_point_provenance='primary_loss_plot_data.csv',validation_loss_not_substituted_by_one_minus_Dice=True,history_reference='original frozen selection-validation reference',final_test_reference='revised reference exactly as archived final paper package',historical_same_position_model_exposure=True,external_independent_test_images=0,original_validation_positive_pixels=npos,original_validation_valid_pixels=nvalid,epoch_counts_verified_against_recorded_metrics=True,learning_rates_saved_each_measured_epoch=True,LR_reconstruction_formula='lr0*(0.1+0.9*0.5*(1+cos(pi*(epoch-1)/epochs_max)))',source_history_and_configuration_hashes=inputs,probability_inputs='source_probability_manifest.json',outputs=outputs,metrics=evaluations)
    js(out/'figure_protocol.json',protocol);js(out/'verification.json',dict(passed=True,preserved_baseline_models_match_old_scores=True,all_new_final_counts_recomputed_from_probabilities=True,all_epoch_counts_reproduce_logged_scores=True,all_AUC_AP_use_continuous_probabilities=True,validation_loss_fabricated=False,additional_training_completed=True,figure_count=len(outputs),DPI=600));js(out/'completion.json',dict(state='complete',historical_loss_origin='Actual replay measurements, not originally recorded Validation Loss',figures=len(outputs)))
    print('ALL_AVAILABLE_FINAL_DIAGNOSTICS_COMPLETE',len(outputs),flush=True)

if __name__=='__main__':main()
