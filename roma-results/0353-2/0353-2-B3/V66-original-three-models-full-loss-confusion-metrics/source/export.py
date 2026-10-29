"""Original V66 models only: fixed-root temporal evaluation, genuine loss logs and figures."""
import csv,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_V66_three_original_models_complete_report_20261029'
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
REF=ROOT/'results/B3_T4_all_extension_reference_to_new_v87_20261028'
PRIOR=ROOT/'results/B3_V90_SCI_two_figures_20261029'
CURVES=ROOT/'results/B3_V90_manual_original_UNet_report_20261029'
PERIODS=['T1','T2','T3','T4'];MODELS={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
CHECKPOINTS={'unet':'58a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382',
             'deeplab':'0b73e59b1928739a48034d128be0eca9778202e90bf20917abd59b1a2ac28d1d',
             'segformer':'cff450a88c0ab2509172eaa828cd9098556eea80550d9ab87089e17af7118f69'}
sys.path.insert(0,str(HERE.parent/'v90_sci_two_figures'))
from export import workbook
sys.path.insert(0,str(HERE.parent/'root_axis_junction_v83'))
from prepare import sha,js,csvout
from rendering import isolated,FONT


def metrics(cm):
    support=cm.sum(1);tp=np.diag(cm[:,:3]);fp=cm[:,:3].sum(0)-tp;fn=support-tp
    precision=tp/(tp+fp);recall=tp/(tp+fn);dice=2*tp/(2*tp+fp+fn);iou=tp/(tp+fp+fn)
    n=int(cm.sum());predicted=int(cm[:,:3].sum());correct=int(tp.sum())
    return dict(Macro_Dice=float(dice.mean()),Macro_IoU=float(iou.mean()),Macro_Precision=float(precision.mean()),Macro_Recall=float(recall.mean()),
                Root_accuracy=correct/n,Micro_Dice=2*correct/(n+predicted),Micro_Precision=correct/predicted,Micro_Recall=correct/n,
                Evaluated_root_pixels=n,Missed_root_pixels=int(cm[:,3].sum()),Correct_root_pixels=correct),dict(support=support.tolist(),TP=tp.tolist(),FP=fp.tolist(),FN=fn.tolist(),Dice=dice.tolist(),IoU=iou.tolist(),Precision=precision.tolist(),Recall=recall.tolist())


def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    for d in ['figures','histories','native_roots','labels','reference','source']:(OUT/d).mkdir(parents=True,exist_ok=True)
    refjson=REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json'
    assert sha(refjson)=='0664587ee2e5931e318f8ad025934b64d5a95fb4ffb8854ac942da861b907636'
    files=[refjson]+[REF/'corrected_reference'/f'{t}_classes.png' for t in PERIODS]
    for m in MODELS:
        files += [BASE/m/n for n in ['best_temporal_head.pt','history.csv','completion.json','config.json']]
        files += [BASE/m/'full'/f'{t}_classes.png' for t in PERIODS]
    source_full='Fig1_Three_model_100_epoch_curves_and_four_period_full_comparison'
    source_loss='Fig5_Original_three_models_complete_100_epoch_training_validation_loss'
    for name in [source_full+'.tif',source_full+'.pdf',source_full+'_preview.png']:files.append(PRIOR/'figures'/name)
    for name in [source_loss+'.tif',source_loss+'.svg',source_loss+'_preview.png']:files.append(CURVES/'figures'/name)
    frozen={str(p):sha(p) for p in files}
    reference={t:np.array(Image.open(REF/'corrected_reference'/f'{t}_classes.png')) for t in PERIODS}
    for t in PERIODS:shutil.copy2(REF/'corrected_reference'/f'{t}_classes.png',OUT/'reference'/f'{t}_classes.png')
    summary=[];allcounts={};cmrows=[];normrows=[];periodrows=[];lossrows=[];runs=[]
    for m,name in MODELS.items():
        assert sha(BASE/m/'best_temporal_head.pt')==CHECKPOINTS[m]
        complete=json.loads((BASE/m/'completion.json').read_text());assert complete['epochs_completed']==100 and complete['checkpoint_sha256']==CHECKPOINTS[m]
        rows=list(csv.DictReader((BASE/m/'history.csv').open(encoding='utf-8-sig')));training=[r for r in rows if int(r['epoch'])>0]
        assert [int(r['epoch']) for r in training]==list(range(1,101))
        assert np.isfinite([[float(r['training_loss']),float(r['validation_loss'])] for r in training]).all()
        for r in training:lossrows.append(dict(Model=name,Epoch=int(r['epoch']),Training_loss=float(r['training_loss']),Validation_loss=float(r['validation_loss'])))
        for f in ['history.csv','config.json','completion.json']:shutil.copy2(BASE/m/f,OUT/'histories'/f'{m}_{f}')
        runs.append(dict(Model=name,Completed_epochs=100,Selected_epoch=complete['selected_epoch'],Checkpoint_SHA256=CHECKPOINTS[m],History_SHA256=sha(BASE/m/'history.csv'),Training_reference='upload18; original run, binary/features frozen'))
        total=np.zeros((3,4),np.int64);periods={}
        (OUT/'native_roots'/m).mkdir(exist_ok=True);(OUT/'labels'/m).mkdir(exist_ok=True)
        for t in PERIODS:
            p=np.array(Image.open(BASE/m/'full'/f'{t}_classes.png'))
            shutil.copy2(BASE/m/'full'/f'{t}_classes.png',OUT/'labels'/m/f'{t}_classes.png')
            Image.fromarray(isolated(p)).save(OUT/'native_roots'/m/f'{t}_roots.png')
            if t=='T1':continue
            g=reference[t];use=np.isin(g,[1,2,3]);assert np.isin(p[use],[0,1,2,3]).all()
            y=g[use].astype(np.int64)-1;z=np.where(p[use]==0,3,p[use]-1)
            cm=np.bincount(y*4+z,minlength=12).reshape(3,4);total+=cm
            periods[t]=cm.tolist()
        assert total.sum()==468203
        mm,classaudit=metrics(total);summary.append(dict(Model=name,**mm))
        norm=total/total.sum(1)[:,None]
        for i,label in enumerate(['Persistent','New','Extension']):
            cmrows.append(dict(Model=name,Manual_class=label,Predicted_persistent=int(total[i,0]),Predicted_new=int(total[i,1]),Predicted_extension=int(total[i,2]),Missed=int(total[i,3]),Support=int(total[i].sum())))
            normrows.append(dict(Model=name,Manual_class=label,Predicted_persistent=float(norm[i,0]),Predicted_new=float(norm[i,1]),Predicted_extension=float(norm[i,2]),Missed=float(norm[i,3])))
        allcounts[m]=dict(model=name,confusion_counts=total.tolist(),row_normalized=norm.tolist(),per_period_confusion=periods,per_class_audit=classaudit,metrics=mm)
    # U-Net row must reproduce the previously reported current-reference foreground score.
    prev=json.loads((CURVES/'metrics.json').read_text())['all_valid_annotated_root_pixels']['T2_T3_T4_combined']['original_U_Net_V66']
    assert np.array_equal(np.array(allcounts['unet']['confusion_counts'])[:,:3],prev['confusion'])
    assert abs(summary[0]['Macro_Dice']-prev['macro_Dice'])<1e-14
    assert len({tuple(np.array(v['confusion_counts']).sum(1)) for v in allcounts.values()})==1
    csvout(OUT/'V66_Three_models_temporal_metrics.csv',summary)
    csvout(OUT/'V66_Confusion_counts.csv',cmrows);csvout(OUT/'V66_Confusion_row_normalized.csv',normrows)
    csvout(OUT/'V66_Actual_100_epoch_losses.csv',lossrows);csvout(OUT/'V66_Original_training_runs.csv',runs)
    js(OUT/'V66_metrics_and_confusion.json',allcounts)
    notes=[dict(Item='Models',Definition='Only original V66 U-Net, DeepLabV3+, SegFormer-B0; no V90/ConvLSTM/GAT output substituted.'),
           dict(Item='Reference',Definition='Current revised manual V87, immutable SHA256 '+sha(refjson)),
           dict(Item='Evaluation scope',Definition='Same468203 manually typed root pixels across T2-T4 for every model. No true-background pixels or binary segmentation score. T1 baseline displayed, not a change class.'),
           dict(Item='Macro metrics',Definition='Arithmetic mean of three temporal foreground-class metrics after pooling counts across dates. Per-date Dice values are not averaged.'),
           dict(Item='Missed',Definition='A manually typed root pixel predicted as0 remains in its reference row and counts as FN; shown in fourth prediction column. It is not a separate background-segmentation evaluation.'),
           dict(Item='Accuracy',Definition='Correct temporal labels divided by468203 reference root pixels, including missed roots.'),
           dict(Item='Original logs',Definition='Actual original upload18 training:100epochs; U-Net37, DeepLab14, SegFormer63 selected by validation macroDice. Current-reference scores are not the old upload18 validation scores.'),
           dict(Item='Limitations',Definition='Single-position development agreement; manual root silhouettes nearly inherit original U-Net foreground. Not independent test or cross-tube generalization.'),
           dict(Item='Changed scope vs earlier root/background table',Definition='Explicit true-background pixels and their FP are excluded at user request. Do not mix these root-only scores with earlier whole-image/binary/4x4-background scores.')]
    workbook(OUT/'V66_Three_models_results.xlsx',[('Metrics',summary),('Confusion_counts',cmrows),('Confusion_normalized',normrows),('Training_runs',runs),('Definitions',notes)])
    # Original previously verified full-field/curves artwork is copied losslessly, not relabeled model output.
    fullstem='Fig1_V66_Three_models_four_period_and_100_epoch_curves'
    lossstem='Fig2_V66_Three_models_100_epoch_training_validation_loss'
    for ext in ['.tif','.pdf','_preview.png']:shutil.copy2(PRIOR/'figures'/(source_full+ext),OUT/'figures'/(fullstem+ext))
    for ext in ['.tif','.svg','_preview.png']:shutil.copy2(CURVES/'figures'/(source_loss+ext),OUT/'figures'/(lossstem+ext))
    font_manager.fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    figdir=OUT/'figures';newfigures=[]
    # A vector PDF counterpart of the unchanged original standalone loss artwork.
    lf,laxes=plt.subplots(1,3,figsize=(18,5.2),constrained_layout=True)
    for ax,(m,name),run in zip(laxes,MODELS.items(),runs):
        rr=[r for r in lossrows if r['Model']==name]
        ax.plot([r['Epoch'] for r in rr],[r['Training_loss'] for r in rr],color='#0072B2',lw=1.5,label='Training loss')
        ax.plot([r['Epoch'] for r in rr],[r['Validation_loss'] for r in rr],color='#D55E00',lw=1.5,label='Validation loss')
        ax.axvline(run['Selected_epoch'],color='#666666',ls='--',lw=1,label=f"Selected epoch {run['Selected_epoch']}")
        ax.set_title(name);ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_xlim(1,100);ax.set_xticks([1,20,40,60,80,100]);ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    lf.suptitle('Original three-model temporal training: 100 actual epochs (V66, reference upload18)',fontsize=16,fontweight='bold')
    lf.savefig(figdir/(lossstem+'.pdf'));plt.close(lf)
    def save(fig,stem):
        tif=figdir/(stem+'.tif');pdf=figdir/(stem+'.pdf');svg=figdir/(stem+'.svg')
        fig.savefig(tif,dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(pdf,dpi=300);fig.savefig(svg)
        im=Image.open(tif).convert('RGB');dims=list(im.size);im.thumbnail((2400,1800));im.save(figdir/(stem+'_preview.png'));plt.close(fig)
        newfigures.append(dict(stem=stem,pixels=dims,tiff_dpi=600));print('FIGURE',stem,flush=True)
    # Row-normalized confusion with raw counts. A shared scale makes models directly comparable.
    fig,axes=plt.subplots(1,3,figsize=(16.2,5.3));fig.subplots_adjust(left=.065,right=.94,bottom=.22,top=.80,wspace=.42)
    for ax,(m,name) in zip(axes,MODELS.items()):
        cm=np.array(allcounts[m]['confusion_counts']);norm=cm/cm.sum(1)[:,None]
        hm=ax.imshow(norm,cmap='Blues',vmin=0,vmax=1,aspect='equal',interpolation='nearest')
        ax.set_xticks(range(4),['Persistent','New','Extension','Missed'],fontsize=9,rotation=25,ha='right')
        ax.set_yticks(range(3),['Persistent','New','Extension'],fontsize=9)
        ax.set_xlabel('Prediction',fontsize=10);ax.set_ylabel('Manual reference',fontsize=10);ax.set_title(name,fontsize=13,pad=12)
        ax.tick_params(length=0,pad=6)
        for i in range(3):
            for j in range(4):ax.text(j,i,f'{cm[i,j]:,}\n{100*norm[i,j]:.1f}%',ha='center',va='center',fontsize=8.3,fontweight='bold',color='white' if norm[i,j]>.5 else '#18212A')
        ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1.5);ax.tick_params(which='minor',length=0)
        for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(hm,cax=fig.add_axes([.959,.30,.010,.40]));cb.set_label('Fraction within reference class',fontsize=9);cb.ax.tick_params(labelsize=8)
    fig.suptitle('V66 original models: pooled temporal confusion matrices (T2-T4)',fontsize=16,y=.96,fontweight='bold')
    fig.text(.5,.035,'Same 468,203 manually typed root pixels | Each cell: count and row percentage | Missed root pixels remain false negatives',ha='center',fontsize=10,fontweight='bold')
    save(fig,'Fig3_V66_Three_models_temporal_confusion_matrices')
    # Metric bars and a compact numeric table in the same publication panel.
    fig=plt.figure(figsize=(13.5,7.2));ax=fig.add_axes([.075,.43,.88,.42]);keys=['Macro_Dice','Macro_IoU','Macro_Precision','Macro_Recall','Root_accuracy'];labels=['Dice','IoU','Precision','Recall','Accuracy'];x=np.arange(5);width=.22
    for i,(row,color) in enumerate(zip(summary,['#0072B2','#D55E00','#009E73'])):
        vals=[row[k] for k in keys];bars=ax.bar(x+(i-1)*width,vals,width,label=row['Model'],color=color)
        for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.015,f'{v:.3f}',ha='center',fontsize=8,fontweight='bold')
    ax.set_ylim(0,1.10);ax.set_yticks(np.arange(0,1.01,.2));ax.set_xticks(x,labels);ax.set_ylabel('Score');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncols=3,loc='upper center',bbox_to_anchor=(.5,1.24),fontsize=10)
    ta=fig.add_axes([.075,.13,.88,.20]);ta.set_axis_off();cell=[[r['Model']]+[f'{r[k]:.4f}' for k in keys] for r in summary]
    table=ta.table(cellText=cell,colLabels=['Model','Macro Dice','Macro IoU','Macro precision','Macro recall','Root accuracy'],loc='center',cellLoc='center',colWidths=[.23]+[.154]*5)
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.6)
    for (i,j),cell in table.get_celld().items():cell.set_facecolor('white');cell.set_edgecolor('#A0A6AB');cell.set_linewidth(.55);cell.get_text().set_fontweight('bold')
    fig.suptitle('V66 original three-model temporal performance',fontsize=16,y=.97,fontweight='bold')
    fig.text(.5,.035,'Root-only manual-reference agreement; T2-T4 pooled. First four metrics are class macro averages; missed roots are retained.',ha='center',fontsize=9,fontweight='bold')
    save(fig,'Fig4_V66_Three_models_temporal_metric_comparison')
    js(OUT/'figure_manifest.json',dict(figures=[dict(stem=fullstem,source_stem=source_full,copied_losslessly=True),dict(stem=lossstem,source_stem=source_loss,copied_losslessly_formats=['TIFF','SVG','previewPNG'],PDF='Vector plot from the same unchanged genuine V66 history')]+newfigures,
        models=list(MODELS.values()),V90_or_CG_predictions_included=False,manual_reference='revisedV87',T1_baseline_display=True,all_100_loss_epochs_actual=True,
        matrix_rows=['Persistent','New','Extension'],matrix_columns=['Persistent','New','Extension','Missed'],matrix_heatmap='row-normalized, common0-1scale; count and percentage annotated',font='black bold embedded Times New Roman'))
    text='''# V66三个原始模型完整结果

本包只包含原始V66 U-Net、DeepLabV3+、SegFormer-B0。没有V90、ConvLSTM或GAT预测替代项，没有重新训练、改预测、改人工标注或根系轮廓。

图1为三模型100轮曲线与四期完整可用上半观察区（2541×1108）对比：RGB、修订人工标注、三个原始模型；分类列含叠加图和白底根系。图2单独给出三个模型完整1–100轮训练与验证损失，无平滑、补轮或模拟。图3为混淆矩阵，图4为综合指标与数值表。另附原尺寸四期各模型彩色根系PNG、原始类别PNG、人工参照和真实训练日志。

## 口径

沿用本轮对话的当前修订人工参照V87，只评价根系上的时序类别，不评价背景分割。三个模型共同评估T2–T4全部468203个人工已标类型的根像素；T1仅基线展示，不是新生/持续/延伸类别，不计入三类指标。先跨期汇总计数，再算指标；Dice/IoU/Precision/Recall为持续、新生、延伸三类宏平均，准确率为正确类别数除以468203。

混淆矩阵行是真实持续/新生/延伸，列是预测三类及“未检出”。预测0的人工根像素计入相应真实类别FN，不丢弃；图中每格同时显示计数与行百分比，三模型共用0–1色标。未检出列不是背景分割指标。明确背景像素不参与本次表格，故此表不能混用此前含背景误检的整图宏平均；也不能把3×3已检出条件矩阵当作完整覆盖结果。

## 来源与限制

权重、预测与训练损失全部为V66原始同批文件：100实际训练轮，选中轮次U-Net37、DeepLabV3+14、SegFormer-B0 63。旧训练使用(18).json、冻结二值分割及特征后训练时序头；选轮按旧验证宏平均Dice。这里的预测重评估使用当前修订V87人工标签，不伪称原(18)训练日志上的验证/测试分数。图像和历史曲线以文件SHA256核验。

人工根系轮廓几乎沿用原始U-Net，当前所有B3数据已作为同位置开发反馈使用，因此这些是人工参照下的开发一致度，不能当作独立测试或跨管泛化排名。三个模型评估支持数完全相同，但其他模型有根漏检，已明确保留。

## 综合指标

| 模型 | Dice | IoU | Precision | Recall | 准确率 | 未检出根像素 |\n|---|---:|---:|---:|---:|---:|---:|\n'''
    for r in summary:text+=f"| {r['Model']} | {r['Macro_Dice']:.6f} | {r['Macro_IoU']:.6f} | {r['Macro_Precision']:.6f} | {r['Macro_Recall']:.6f} | {r['Root_accuracy']:.6f} | {r['Missed_root_pixels']} |\n"
    text+='\nExcel含指标、原始/归一化混淆矩阵、训练来源和定义页。PDF及表格日志包便于统一下载；600dpi TIFF可单独下载。\n'
    (OUT/'RESULTS.md').write_text(text)
    (OUT/'Figure_captions.txt').write_text('Figure1. Original V66 three-model100-epoch training dynamics and four-period full-field RGB/manual/original-model comparison. Training used upload18, displayed reference is revisedV87.\nFigure2. Actual original V66 training and validation losses for all100epochs; validation-macro-Dice-selected epochs37/14/63.\nFigure3. Temporal confusion over the same468203 manually typed root pixels in T2-T4. Rows are reference classes and columns predicted classes plus missed roots. Counts and row percentages are annotated; shared0-1scale. T1baseline and true background pixels are excluded.\nFigure4. Root-only pooled macro Dice/IoU/Precision/Recall and root-label accuracy for the three original V66 models. Missed roots remain FN. All results are same-position development-reference agreement, not independent test evidence.\n')
    for p,h in frozen.items():assert sha(p)==h
    js(OUT/'verification.json',dict(passed=True,only_original_V66_models=True,immutable_input_sha256=frozen,all_three_original_checkpoint_SHA256_verified=True,
        all_actual_100_epoch_histories_verified=True,same_root_support_across_models=468203,manual_reference_unchanged=True,predictions_unchanged=True,
        no_training=True,no_background_segmentation_metrics=True,missed_roots_retained_in_confusion_and_FN=True,original_UNet_row_matches_previous_root_only_score=True,
        original_source_curves_and_full_figure_copied_losslessly=True,not_independent_test=True))
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    js(OUT/'completion.json',dict(state='complete',models=3,figures=4,epochs_per_model=100,evaluated_root_pixels_per_model=468203,no_training=True))
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','download_links.json'] and not p.name.startswith(('published_','publication')) and p.suffix not in ['.zip','.log']]
    package=OUT/'V66_PDF_tables_and_logs.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            if 'figures' in p.relative_to(OUT).parts and p.suffix!='.pdf':continue
            z.write(p,p.relative_to(OUT))
    with zipfile.ZipFile(package) as z:assert z.testzip() is None
    assert package.stat().st_size<24_000_000
    js(OUT/'artifact_manifest.json',[dict(file=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in files+[package]])
    print('COMPLETE',json.dumps(summary),package.stat().st_size,flush=True)


if __name__=='__main__':main()
