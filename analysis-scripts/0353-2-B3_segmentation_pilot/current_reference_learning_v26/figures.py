"""Publication composites using the current reference for every displayed score."""
import argparse, hashlib, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import common as c

COLORS=['#0072B2','#D55E00','#009E73']


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--trajectory',type=Path,default=Path('results/B3_binary_continuation_v25_20261021'))
    parser.add_argument('--table-only',action='store_true',help='Redraw from delivered current-reference tables; no model/data loading')
    args=parser.parse_args();out=args.run/'report';out.mkdir(exist_ok=True);folder=out/'figures';folder.mkdir(exist_ok=True)
    curves={};combined=[];inputs=[]
    delivered_rows=c.csvread(out/'all_current_reference_epochs.csv') if args.table_only else None
    prior_protocol=c.read(out/'figure_protocol.json') if (out/'figure_protocol.json').exists() else None
    for model in c.MODELS:
        if args.table_only:
            hp=out/'all_current_reference_epochs.csv';rows=[{key:value for key,value in r.items() if key!='model'} for r in delivered_rows if r['model']==model]
        else:
            assert c.read(args.run/model/'completion.json')['state']=='complete'
            hp=args.run/model/'history.csv';rows=c.csvread(hp)
        assert len(rows)==30 and [int(r['epoch']) for r in rows]==list(range(1,31))
        for row in rows:
            assert all(np.isfinite(float(row[key])) for key in ['train_loss','validation_loss','validation_Dice','validation_IoU'])
            tp,fp,fn,tn=[int(row['validation_'+key]) for key in ['TP','FP','FN','TN']]
            assert abs(float(row['validation_Dice'])-2*tp/(2*tp+fp+fn))<1e-12
            assert abs(float(row['validation_IoU'])-tp/(tp+fp+fn))<1e-12
            combined.append(dict(model=model,**{key:value for key,value in row.items() if not key.startswith('audit_') and key!='train_loss_difference_from_V25'}))
        curves[model]={key:np.array([float(r[key]) for r in rows]) for key in rows[0] if key not in ['phase']}
        inputs.append(dict(model=model,path=str(hp),SHA256=c.sha(hp),reference_manifest_SHA256=c.sha(args.run/'data/manifest.csv')))
    c.csvout(out/'all_current_reference_epochs.csv',combined)
    scores=c.csvread(args.run/'retained_models_current_reference_scores.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'grid.color':'#E5E9ED','grid.linewidth':.5,'savefig.facecolor':'white','pdf.fonttype':42})
    outputs=[];captions={};profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    def save(fig,name,caption):
        preview=out/(name+'_preview.png');temporary=out/(name+'_600.png')
        fig.savefig(preview,dpi=180);fig.savefig(temporary,dpi=600);plt.close(fig)
        path=folder/(name+'.tif')
        with Image.open(temporary) as image:image.convert('RGB').save(path,compression='tiff_lzw',dpi=(600,600),icc_profile=profile)
        with Image.open(path) as image:
            image.load();assert image.mode=='RGB' and image.n_frames==1 and tuple(image.info['dpi'])==(600.,600.);size=list(image.size)
        temporary.unlink();outputs.append(dict(file=path.name,bytes=path.stat().st_size,SHA256=c.sha(path),size=size,DPI=600));captions[name]=caption;print('SAVED_CURRENT_REFERENCE_FIGURE',path.name,flush=True)
    def decorate(ax):
        ax.grid(alpha=.6);ax.set_axisbelow(True);ax.set_xlim(0,30.5);ax.xaxis.set_major_locator(MultipleLocator(5));ax.axvline(20.5,color='#9BA1A8',ls=':',lw=.8);ax.tick_params(length=3,width=.65)
    lossmax=max(max(q['train_loss'].max(),q['validation_loss'].max()) for q in curves.values())*1.13
    def loss_panel(ax,model):
        q=curves[model];ax.plot(q['epoch'],q['train_loss'],color='#0072B2',lw=1.2,label='Training loss');ax.plot(q['epoch'],q['validation_loss'],color='#D55E00',lw=1.2,label='Validation loss');decorate(ax);ax.set(xlabel='Epoch',ylabel='Loss',ylim=(0,lossmax));ax.legend(frameon=False,fontsize=6.5,loc='upper right')
    footer='Current reference · measured replay · dotted line: checkpoint continuation'
    loss_caption='All30measured epochs for each of the three models; each epoch is32minibatches of2, or64augmented training draws (not a full sweep of the200views). Training targets and validation targets use the same user-confirmed current annotation version. Loss is hard-negative BCE+softDice+0.15softclDice; training batches are augmented, validation uses the full20view unaugmented validation set in evaluation mode. Epochs1–20replay the established training path; the reported final post-annotation-training checkpoint is reloaded at the dotted boundary and epochs21–30replay the recorded continuation. Per-epoch rates and image draws are frozen to the recorded trajectory. These are actual newly measured replay losses, not recovered original validation logs. No curve is interpolated, smoothed or forced to converge. The already reported final checkpoints and test scores are unchanged.'
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.85));fig.subplots_adjust(left=.075,right=.985,top=.80,bottom=.25,wspace=.38)
    for index,(ax,model) in enumerate(zip(axes,c.MODELS)):
        loss_panel(ax,model);ax.set_title(c.NAMES[model],pad=9);ax.text(-.15,1.09,chr(97+index),transform=ax.transAxes,fontweight='bold',fontsize=10)
    fig.suptitle('Three-model training and validation',fontsize=10,fontweight='semibold',y=.98);fig.text(.53,.035,footer,ha='center',fontsize=6.1,color='#5F6873')
    save(fig,'Fig01_three_model_training_validation_loss',loss_caption)
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.2));fig.subplots_adjust(left=.075,right=.985,top=.87,bottom=.16,wspace=.38,hspace=.55)
    for index,model in enumerate(c.MODELS):
        loss_panel(axes[0,index],model);axes[0,index].set_title(c.NAMES[model],pad=10)
        q=curves[model];ax=axes[1,index];ax.plot(q['epoch'],q['validation_Dice'],color='#0072B2',lw=1.2,marker='o',ms=2,label='Validation Dice');ax.plot(q['epoch'],q['validation_IoU'],color='#D55E00',lw=1.2,marker='o',ms=2,label='Validation IoU');decorate(ax);ax.set(xlabel='Epoch',ylabel='Score',ylim=(0,1.02));ax.legend(frameon=False,fontsize=6.4,loc='lower left')
        for row in range(2):axes[row,index].text(-.15,1.08,chr(97+row*3+index),transform=axes[row,index].transAxes,fontweight='bold',fontsize=10)
    fig.suptitle('Learning dynamics with the current annotation reference',fontsize=10,fontweight='semibold',y=.98);fig.text(.53,.045,footer,ha='center',fontsize=6.1,color='#5F6873')
    save(fig,'Fig02_three_model_learning_dynamics_composite',loss_caption+' Lower panels show validation Dice and IoU on the current reference at the fixed model-specific thresholds0.45,0.40and0.475. Validation scores are not the final test-set scores and are not forced to equal them.')
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.85));fig.subplots_adjust(left=.085,right=.985,top=.80,bottom=.24,wspace=.40)
    for index,(ax,model,color) in enumerate(zip(axes,c.MODELS,COLORS)):
        q=curves[model];ax.plot(q['epoch'],q['learning_rate'],color=color,lw=1.3);decorate(ax);ax.set(title=c.NAMES[model],xlabel='Epoch',ylabel='Learning rate',ylim=(0,9e-5));ax.ticklabel_format(axis='y',style='sci',scilimits=(-5,-5),useMathText=True)
    fig.suptitle('Recorded learning-rate schedules',fontsize=10,fontweight='semibold',y=.98);fig.text(.53,.035,'Rates follow the recorded training trajectory; validation targets use the current reference',ha='center',fontsize=6.3,color='#5F6873')
    save(fig,'Fig03_three_model_learning_rate_schedules','Actual rates used in the measured replay. Rates are frozen at their recorded values so that changing the monitoring reference cannot change the established training path. The dotted line marks checkpoint restoration and optimizer restart. Every epoch is shown.')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.1));fig.subplots_adjust(left=.085,right=.985,top=.88,bottom=.17,wspace=.25,hspace=.40)
    keys=['validation_Dice','validation_IoU','validation_Accuracy','validation_Balanced_accuracy'];labels=['Validation Dice','Validation IoU','Pixel accuracy','Balanced accuracy']
    for index,(ax,key,label) in enumerate(zip(axes.ravel(),keys,labels)):
        for model,color in zip(c.MODELS,COLORS):ax.plot(curves[model]['epoch'],curves[model][key],color=color,lw=1.15,marker='o',ms=2.3,label=c.NAMES[model])
        decorate(ax);ax.set(xlabel='Epoch',ylabel=label,ylim=(0,1.02));ax.text(-.12,1.05,chr(97+index),transform=ax.transAxes,fontweight='bold',fontsize=10)
    fig.suptitle('Validation metrics · current annotation reference',fontsize=10,fontweight='semibold',y=.98);fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.03),ncol=3,frameon=False,fontsize=7)
    save(fig,'Fig04_current_reference_validation_metrics','All validation curves use only the current annotation reference on20validation views at5spatial coordinates. Model thresholds are fixed and identical to those used for the final test reporting. Dice,IoU,pixel accuracy and balanced accuracy are recomputed from saved integer pixel counts. No original-reference validation scores are plotted. The curves are measured replay records and are not inferred from final test values.')
    fig,ax=plt.subplots(figsize=(7.2,3.2));fig.subplots_adjust(left=.095,right=.985,top=.83,bottom=.25);x=np.arange(3);width=.30
    for split,offset,color,label in [('val',-.5,'#D55E00','Validation (20 views)'),('test',.5,'#0072B2','Test (24 views)')]:
        values=[float(next(r for r in scores if r['model']==model and r['split']==split)['Dice']) for model in c.MODELS]
        xx=x+offset*width;ax.bar(xx,values,width*.93,color=color,label=label)
        for xx1,value in zip(xx,values):ax.text(xx1,value+.016,'%.4f'%value,ha='center',fontsize=8)
    ax.set_xticks(x,[c.NAMES[model] for model in c.MODELS]);ax.set(ylim=(0,1.05),ylabel='Dice',title='Reported final models · validation and test');ax.grid(axis='y',alpha=.5);ax.set_axisbelow(True);fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.53,.055),ncol=2,frameon=False,fontsize=7.5)
    save(fig,'Fig05_final_model_validation_test_comparison','The same retained reported final checkpoints are scored on validation20views and test24views using one current annotation version and the same model-specific fixed thresholds. Only the split changes. Test Dice remains0.8845978,0.7332927and0.7255331. Corresponding validation Dice is0.8133791,0.6673410and0.5937614. These are selected-checkpoint scores, not a claim that every epoch achieves the final test score. This same-position evaluation does not establish external tube generalization.')
    reused=[('Fig04_final_test_metric_comparison.tif','Fig06_final_test_metric_comparison.tif'),('Fig05_final_test_confusion_matrices.tif','Fig07_final_test_confusion_matrices.tif'),('Fig06_final_test_ROC_AUC_and_PR.tif','Fig08_final_test_ROC_AUC_and_PR.tif')]
    for old,name in reused:
        original=args.trajectory/'report/figures'/old;target=folder/name
        if original.exists():
            shutil.copy2(original,target);assert c.sha(target)==c.sha(original)
        else:
            assert args.table_only and prior_protocol is not None
            expected=next(r['SHA256'] for r in prior_protocol['outputs'] if r['file']==name)
            assert target.is_file() and c.sha(target)==expected,'The delivered fixed test TIFF is missing or changed'
        with Image.open(target) as image:image.load();assert image.mode=='RGB' and tuple(image.info['dpi'])==(600.,600.);size=list(image.size)
        outputs.append(dict(file=name,bytes=target.stat().st_size,SHA256=c.sha(target),size=size,DPI=600,unchanged_current_reference_test_result=True));captions[name.removesuffix('.tif')]='The previously verified current-reference test result is reused byte-for-byte. Reported final checkpoints, current test masks, probabilities and thresholds are unchanged. No original-reference score is substituted.'
    c.js(out/'captions.json',captions)
    c.js(out/'figure_protocol.json',dict(current_reference_only_in_figures=True,reference_manifest_SHA256=c.sha(args.run/'data/manifest.csv'),curves_are_new_measured_replay=True,epochs_each=30,phase_boundary=20.5,training_views=200,validation_views=20,test_views=24,source_original_RGB_images=4,external_independent_test_images=0,reported_final_models_unchanged=True,reported_final_model_scores='../retained_models_current_reference_scores.csv',histories=inputs,outputs=outputs,old_reference_metrics_displayed=False))
    c.js(out/'completion.json',dict(state='complete',figures=len(outputs),current_reference_only=True))
    print('ALL_CURRENT_REFERENCE_FIGURES_COMPLETE',len(outputs),flush=True)


if __name__=='__main__':main()
