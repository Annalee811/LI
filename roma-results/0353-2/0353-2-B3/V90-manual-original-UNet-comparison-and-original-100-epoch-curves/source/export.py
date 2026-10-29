"""Immutable V90/manual/original-U-Net comparison and the actual V66 loss logs."""
import csv, hashlib, json, shutil, sys, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/workspace/LI')
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/B3_V90_manual_original_UNet_report_20261029'
BASE = ROOT / 'results/B3_temporal_upload18_noGAT_v66_20261028'
CURRENT = ROOT / 'results/B3_user_confirmed_new_axis_correction_20261029'
REF = ROOT / 'results/B3_T4_all_extension_reference_to_new_v87_20261028'
SPATIAL = ROOT / 'results/B3_original_UNet_latestCG_upload24_v82_20261028/data/spatial.npy'
PERIODS = ['T1', 'T2', 'T3', 'T4']
MODEL_NAMES = {'unet': 'U-Net', 'deeplab': 'DeepLabV3+', 'segformer': 'SegFormer-B0'}
sys.path.insert(0, str(ROOT / 'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from rendering import Canvas, isolated, overlay, FONT


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def js(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csvout(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def score(y, p):
    assert np.isin(y, [1, 2, 3]).all() and np.isin(p, [1, 2, 3]).all()
    cm = np.bincount((y.astype(np.int64)-1)*3 + p.astype(np.int64)-1, minlength=9).reshape(3, 3)
    rows = []
    for k, name in enumerate(['persistent', 'newly_observed', 'tip_extension']):
        tp = int(cm[k, k]); support = int(cm[k].sum()); pred = int(cm[:, k].sum())
        fp = pred-tp; fn = support-tp; den = 2*tp+fp+fn
        rows.append(dict(name=name, Dice=2*tp/den if den else None,
                         Precision=tp/pred if pred else None, Recall=tp/support if support else None,
                         IoU=tp/(tp+fp+fn) if tp+fp+fn else None, support=support,
                         predicted_pixels=pred, TP=tp, FP=fp, FN=fn,
                         status='reference_present' if support else ('absent_reference_false_positive' if pred else 'absent_both_undefined')))
    return dict(classes=rows, accuracy=float(np.trace(cm)/cm.sum()), pixels=int(cm.sum()),
                macro_Dice=float(np.mean([r['Dice'] for r in rows if r['Dice'] is not None])),
                macro_Dice_reference_present_classes=float(np.mean([r['Dice'] for r in rows if r['support']])),
                confusion=cm.tolist())


def fitpanel(cv, arr, x, y, w, h, categorical):
    hh, ww = arr.shape[:2]; scale = min(w/ww, h/hh)
    pw, ph = round(ww*scale), round(hh*scale)
    cv.panel(arr, x+(w-pw)//2, y+(h-ph)//2, pw, ph, categorical)


def main():
    sys.path.insert(0, str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3') == '0353-2-B3'
    for d in ['figures', 'histories', 'source', 'data/reference', 'data/original_UNet', 'data/V90']:
        (OUT/d).mkdir(parents=True, exist_ok=True)
    reference_json = REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json'
    assert sha(reference_json) == '0664587ee2e5931e318f8ad025934b64d5a95fb4ffb8854ac942da861b907636'
    assert sha(BASE/'unet/best_temporal_head.pt') == '58a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382'
    assert json.loads((CURRENT/'verification.json').read_text())['other_root_pixels_changed'] == 0
    inputs = [reference_json, SPATIAL, CURRENT/'verification.json', CURRENT/'Reviewed_connection_and_birth_supervision.json']
    for m in MODEL_NAMES:
        inputs += [BASE/m/n for n in ['history.csv', 'config.json', 'completion.json', 'best_temporal_head.pt']]
    for t in PERIODS:
        inputs += [BASE/'unet/full'/f'{t}_classes.png', CURRENT/'full'/f'{t}_classes.png',
                   REF/'corrected_reference'/f'{t}_classes.png', BASE/'data/upper'/f'{t}_RGB.png']
    frozen = {str(p): sha(p) for p in inputs}
    rgb = {t: np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')) for t in PERIODS}
    gt = {t: np.array(Image.open(REF/'corrected_reference'/f'{t}_classes.png')) for t in PERIODS}
    old = {t: np.array(Image.open(BASE/'unet/full'/f'{t}_classes.png')) for t in PERIODS}
    new = {t: np.array(Image.open(CURRENT/'full'/f'{t}_classes.png')) for t in PERIODS}
    for t in PERIODS:
        assert np.array_equal(old[t]>0, new[t]>0)
        for name, array in [('reference', gt[t]), ('original_UNet', old[t]), ('V90', new[t])]:
            Image.fromarray(array).save(OUT/'data'/name/f'{t}_classes.png')
    assert np.array_equal(old['T1'], new['T1'])
    shutil.copy2(SPATIAL, OUT/'data/spatial.npy')
    shutil.copy2(CURRENT/'Reviewed_connection_and_birth_supervision.json', OUT/'V90_reviewed_correction_provenance.json')
    shutil.copy2(CURRENT/'correction_audit.json', OUT/'V90_single_root_correction_audit.json')
    # Exactly the same reference and pixel supports for both predictions.
    spatial = np.load(SPATIAL); metrics = {}; flat = []; summary = []
    for scope in ['all_valid_annotated_root_pixels', 'original_inner_spatial_audit']:
        metrics[scope] = {}
        for period in PERIODS[1:] + ['T2_T3_T4_combined']:
            dates = PERIODS[1:] if period.endswith('combined') else [period]
            ys=[]; preds={'original_U_Net_V66': [], 'V90_reviewed_output': []}
            for t in dates:
                use=np.isin(gt[t], [1,2,3]) & (old[t]>0)
                if scope=='original_inner_spatial_audit': use &= spatial==2
                ys.append(gt[t][use]); preds['original_U_Net_V66'].append(old[t][use]); preds['V90_reviewed_output'].append(new[t][use])
            y=np.concatenate(ys); metrics[scope][period]={}
            for model, pp in preds.items():
                mm=score(y, np.concatenate(pp)); metrics[scope][period][model]=mm
                for r in mm['classes']: flat.append(dict(scope=scope, period=period, prediction=model, reference='unchanged_user_corrected_V87', **r))
                summary.append(dict(scope=scope, period=period, prediction=model, accuracy=mm['accuracy'],
                                    persistent_Dice=mm['classes'][0]['Dice'], new_Dice=mm['classes'][1]['Dice'],
                                    extension_Dice=mm['classes'][2]['Dice'], macro_Dice=mm['macro_Dice'],
                                    macro_Dice_reference_present_classes=mm['macro_Dice_reference_present_classes'], evaluated_pixels=mm['pixels']))
    # Recomputed values must agree with already published, immutable quantitative outputs.
    previous90=json.loads((CURRENT/'evaluation_results.json').read_text())
    previous66=json.loads((ROOT/'results/B3_original_UNet_vs_V89_comparison_20261029/metrics.json').read_text())
    for scope in metrics:
        for key, prev in [('V90_reviewed_output',previous90[scope]['V90_after_explicit_user_review']),
                          ('original_U_Net_V66',previous66[scope]['original_U_Net_V66'])]:
            curr=metrics[scope]['T2_T3_T4_combined'][key]
            assert np.array_equal(curr['confusion'],prev['confusion'])
            assert np.allclose([r['Dice'] for r in curr['classes']], [r['Dice'] for r in prev['classes']], atol=1e-14,rtol=0)
    js(OUT/'metrics.json', metrics); csvout(OUT/'Accuracy_and_Dice_summary.csv', summary)
    csvout(OUT/'Per_class_precision_recall_Dice_IoU.csv', flat)
    figdir=OUT/'figures'; figs=[]
    cols=[180,1690,3200,4710]; pw=1360; ph=round(pw*1108/2541)
    labels=['Original RGB','Manual reference (revised V87)','Original U-Net (V66)','V90 reviewed output']
    # Overlay and root-only views for every period; full observed upper ROI, no cherry-picked crop.
    cv=Canvas(6250,5780,figdir)
    cv.text(3125,25,'Manual annotation | Original U-Net | V90: full four-period comparison',72)
    cv.text(3125,145,'Same registered full observation area (2541 x 1108 native pixels); original and V90 root geometry is identical',37)
    for x,label in zip(cols,labels): cv.text(x+pw//2,245,label,40)
    for i,t in enumerate(PERIODS):
        top=340+i*1300; cv.text(40,top+530,t,64,'lt')
        fitpanel(cv,rgb[t],cols[0],top+ph//2,pw,ph,False)
        for x,a in zip(cols[1:],[gt[t],old[t],new[t]]):
            cv.panel(overlay(rgb[t],a),x,top,pw,ph)
            cv.panel(isolated(a),x,top+ph+25,pw,ph,True)
    cv.text(3125,5575,'V90 includes the explicit T2 root 0002-N1 review. Manual labels are unchanged. T1 is baseline, excluded from temporal Dice.',33)
    cv.legend(5650); figs.append(cv.save('Fig1_Manual_Original_UNet_V90_four_period_full_comparison'))
    print('FIGURE full overview complete',flush=True)
    # Individual period panels at native resolution, available separately for zoom inspection.
    for t in PERIODS:
        cv=Canvas(10540,1510,figdir); cv.text(5270,20,f'{t}: manual annotation, original U-Net and V90 (native image pixels)',72)
        for x,label,a in zip([90,2700,5310,7920],labels,[rgb[t],isolated(gt[t]),isolated(old[t]),isolated(new[t])]):
            cv.text(x+1270,155,label,55);cv.panel(a,x,270,2541,1108,x!=90)
        cv.legend(1430); figs.append(cv.save(f'Full_{t}_native_resolution_comparison'))
    # Same requested new-root region in all four dates, displaying actual earlier/later evidence.
    matched=[1220,150,2120,520]; x0,y0,x1,y1=matched
    cv=Canvas(6250,3420,figdir);cv.text(3125,25,'Matched four-period detail: the reviewed new lateral 0002-N1',76)
    cv.text(3125,150,'Identical native ROI [1220,150,2120,520] in T1-T4; T2 is explicitly user-reviewed, later dates are preserved',38)
    for x,label in zip(cols,labels):cv.text(x+pw//2,255,label,40)
    for i,t in enumerate(PERIODS):
        top=350+i*680;cv.text(35,top+250,t,62,'lt')
        for j,(x,a) in enumerate(zip(cols,[rgb[t],isolated(gt[t]),isolated(old[t]),isolated(new[t])])):
            fitpanel(cv,a[y0:y1,x0:x1],x,top,pw,570,j!=0)
    cv.text(3125,3180,'Newly observed is a within-period state: a root first seen in T2 may correctly become persistent or extend in a later period.',34)
    cv.legend(3280);figs.append(cv.save('Fig2_Matched_four_period_local_manual_Original_UNet_V90'))
    cases=[('T2',[1220,150,2120,520],'Entire reviewed new root 0002-N1; old contact strand retained'),
           ('T2',[620,740,1068,1100],'Established-root shaft, distal extension and crossing continuity'),
           ('T2',[1290,150,1540,470],'Earlier-period evidence for an established growing root'),
           ('T2',[1530,590,1830,850],'New lateral identity and continuation across older roots'),
           ('T3',[1005,550,1165,715],'Two older roots: crossing ownership and distal growth'),
           ('T3',[1100,205,1290,385],'Different roots at a contact: do not merge their identities')]
    for page in range(2):
        cv=Canvas(6250,3900,figdir);cv.text(3125,25,f'Local temporal / connection details: manual, original U-Net and V90 ({page+1}/2)',71)
        cv.text(3125,155,'User-reported regions, selected before score inspection; identical crops and unchanged root segmentation',38)
        for i,(t,roi,description) in enumerate(cases[page*3:(page+1)*3]):
            top=300+i*1120;x0,y0,x1,y1=roi;cv.text(3125,top,f'{chr(65+page*3+i)}  {t}  {description}',43)
            for j,(x,label,a) in enumerate(zip(cols,labels,[rgb[t],isolated(gt[t]),isolated(old[t]),isolated(new[t])])):
                cv.text(x+pw//2,top+90,label,39)
                fitpanel(cv,a[y0:y1,x0:x1],x,top+165,pw,800,j!=0)
            changed=int((old[t][y0:y1,x0:x1]!=new[t][y0:y1,x0:x1]).sum())
            cv.text(3125,top+995,f'Native ROI {roi}; original-to-V90 changed class pixels: {changed}',35)
        cv.legend(3770);figs.append(cv.save(f'Fig{3+page}_Local_optimization_details_page_{page+1}'))
    print('FIGURE local comparisons complete',flush=True)
    # Actual original-run losses: no interpolation, fabricated epochs, replay or smoothing.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT));plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','font.size':12})
    fig,axs=plt.subplots(1,3,figsize=(18,5.2),constrained_layout=True)
    histories=[]; provenance=[]
    def plot_loss(ax, rows, model, selected):
        ep=[int(r['epoch']) for r in rows]
        ax.plot(ep,[float(r['training_loss']) for r in rows],color='#0072B2',lw=1.55,label='Training loss')
        ax.plot(ep,[float(r['validation_loss']) for r in rows],color='#D55E00',lw=1.55,label='Validation loss')
        ax.axvline(selected,color='#555555',ls='--',lw=1.1,label=f'Selected epoch {selected}')
        ax.set_title(MODEL_NAMES[model]);ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_xlim(1,100)
        ax.set_xticks([1,20,40,60,80,100]);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    for ax,(m,name) in zip(axs,MODEL_NAMES.items()):
        raw=list(csv.DictReader((BASE/m/'history.csv').open(encoding='utf-8-sig')))
        rows=[r for r in raw if int(r['epoch'])>=1]; epochs=[int(r['epoch']) for r in rows]
        assert epochs==list(range(1,101)) and all(r['phase']=='training' for r in rows)
        assert np.isfinite([[float(r['training_loss']),float(r['validation_loss'])] for r in rows]).all()
        complete=json.loads((BASE/m/'completion.json').read_text()); assert complete['epochs_completed']==100
        assert sha(BASE/m/'best_temporal_head.pt')==complete['checkpoint_sha256']
        for f in ['history.csv','config.json','completion.json']:shutil.copy2(BASE/m/f,OUT/'histories'/f'{m}_{f}')
        histories.extend(dict(model=name,epoch=int(r['epoch']),training_loss=float(r['training_loss']),validation_loss=float(r['validation_loss']),selected_epoch=int(complete['selected_epoch'])) for r in rows)
        provenance.append(dict(model=name,source_run='V66 upload18 temporal-head training with frozen binary/features',
                               source_history=str(BASE/m/'history.csv'),source_history_sha256=sha(BASE/m/'history.csv'),
                               epochs=epochs,epoch_zero_preserved_in_raw_csv_but_excluded_from_100_epoch_plot=True,
                               selected_epoch=complete['selected_epoch'],selection_criterion='maximum validation end-to-end macro Dice, not minimum validation loss',
                               checkpoint_sha256=complete['checkpoint_sha256'],curve_values_unmodified=True))
        plot_loss(ax,rows,m,complete['selected_epoch'])
        single,aa=plt.subplots(figsize=(7.5,5),constrained_layout=True);plot_loss(aa,rows,m,complete['selected_epoch'])
        single.suptitle('Original V66 temporal training: all 100 epochs',fontsize=13,fontweight='bold')
        stem=f'Loss_{m}_complete_100_epochs';single.savefig(figdir/(stem+'.tif'),dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});single.savefig(figdir/(stem+'.svg'))
        im=Image.open(figdir/(stem+'.tif')).convert('RGB');im.thumbnail((1800,1900));im.save(figdir/(stem+'_preview.png'));plt.close(single)
        figs.append(dict(file=stem+'.tif',sha256=sha(figdir/(stem+'.tif')),dimensions=list(Image.open(figdir/(stem+'.tif')).size)))
    fig.suptitle('Original three-model temporal training: 100 actual epochs (V66, reference upload18)',fontsize=17,fontweight='bold')
    stem='Fig5_Original_three_models_complete_100_epoch_training_validation_loss'
    fig.savefig(figdir/(stem+'.tif'),dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(figdir/(stem+'.svg'))
    im=Image.open(figdir/(stem+'.tif')).convert('RGB');dims=list(im.size);im.thumbnail((2400,1900));im.save(figdir/(stem+'_preview.png'));plt.close(fig)
    figs.append(dict(file=stem+'.tif',sha256=sha(figdir/(stem+'.tif')),dimensions=dims))
    csvout(OUT/'Three_original_models_actual_100_epoch_losses.csv',histories);js(OUT/'training_curve_provenance.json',provenance)
    # Reference is a standard, not a separately measured model with an asserted 100% accuracy.
    cv=Canvas(5400,2400,figdir);cv.text(2700,30,'Same-reference temporal accuracy and Dice: original U-Net vs V90',68)
    cv.text(2700,145,'Reference: revised manual V87 | T2-T4 combined | T1 baseline excluded | Root foreground fixed',38)
    for i,scope in enumerate(metrics):
        top=310+i*810; heading='All valid annotated root pixels (development agreement)' if i==0 else 'Original inner spatial audit (development region)'
        cv.text(2700,top,heading,46)
        xpos=[170,2450,3050,3650,4250,4850]
        for x,s in zip(xpos,['Variant','Pixel accuracy','Persistent Dice','New Dice','Extension Dice','Macro Dice']):cv.text(x,top+110,s,35,'lt' if x==170 else 'mt')
        for j,(label,mm) in enumerate([('Manual annotation (reference)',None)]+[(k,v) for k,v in metrics[scope]['T2_T3_T4_combined'].items()]):
            yy=top+230+j*115
            cv.text(xpos[0],yy,label,38,'lt')
            vals=['Reference']*5 if mm is None else [f"{mm['accuracy']:.4f}"]+[f"{r['Dice']:.4f}" for r in mm['classes']]+[f"{mm['macro_Dice']:.4f}"]
            for x,s in zip(xpos[1:],vals):cv.text(x,yy,s,38)
        count=metrics[scope]['T2_T3_T4_combined']['original_U_Net_V66']['pixels'];cv.text(2700,top+650,f'Identical evaluated pixels for both outputs: {count:,}',35)
    cv.text(2700,2080,'V90 includes an explicit annotation-assisted correction of T2 root 0002-N1; these are descriptive agreement scores.',36)
    cv.text(2700,2170,'The reference is not an independently evaluated model. Binary segmentation misses outside fixed foreground are excluded.',35)
    figs.append(cv.save('Fig6_Manual_Original_UNet_V90_accuracy_Dice_table'))
    js(OUT/'figure_manifest.json',dict(files=figs,dpi=600,manual_reference_sha256=sha(reference_json),matched_four_period_ROI=matched,detail_cases=[dict(period=t,ROI=r,reason=s) for t,r,s in cases],categorical_resampling='nearest',root_geometry_unchanged=True,full_view='Entire available registered upper observation ROI, not the full tube'))
    for p,h in frozen.items():assert sha(p)==h
    js(OUT/'verification.json',dict(passed=True,immutable_input_sha256=frozen,no_training_or_prediction_edit=True,
                                  reference_unchanged=True,original_V66_checkpoint_verified=True,
                                  V90_exact_published_output=True,root_foreground_identical=True,
                                  same_reference_and_pixel_supports=True,recomputed_metrics_match_published=True,
                                  all_three_histories_have_actual_epochs_1_through_100=True,
                                  original_histories_copied_byte_for_byte=True,curve_source_reference='upload18, not relabeled upload24/V87',
                                  reviewed_output_not_independent_generalization=True))
    lines=['# V90、人工标注与原始 U-Net 对比\n',
           '本包只评估与导出，未训练、未修改任何预测、人工标注或根系轮廓。原始 U-Net 为 V66（100轮，验证集宏平均 Dice 选择第37轮），不是 V75 加入 CG 后的版本。V90 为已发布的优化结果，包含你确认的 T2 根 0002-N1 人工复核修正，不能作为独立模型泛化精度。\n',
           '人工参照固定为 V87：保留你授权的两处 T2 延伸改新生，以及全部 T4 延伸改新生；没有为本次比较再次改标签。人工标注是参照，不作为准确率100%的第三个模型。\n',
           '## 指标\n',
           '两输出使用相同人工标注、相同像素范围。以下是 T2–T4 时序分类，T1 仅基线展示。全部有效标注区包括训练区；原内部空间审计区也已用于反复反馈开发。均为同位置标注一致度，不是独立测试精度。指标限定在冻结的预测根系前景，不含前景之外的分割漏检。\n',
           '| 范围 | 版本 | 像素准确率 | 持续 Dice | 新生 Dice | 延伸 Dice | 宏平均 Dice |\n|---|---|---:|---:|---:|---:|---:|\n']
    for scope,dd in metrics.items():
        for name,mm in dd['T2_T3_T4_combined'].items():lines.append('| '+scope+' | '+name+f" | {mm['accuracy']:.6f} | "+' | '.join(f"{r['Dice']:.6f}" for r in mm['classes'])+f" | {mm['macro_Dice']:.6f} |\n")
    lines += ['\nCSV/JSON 同时给出各期及 T2–T4 汇总的 Dice、Precision、Recall、IoU、TP/FP/FN、类别支持数与混淆矩阵。T4 人工延伸已全部改新生，故延伸类无参照支持：有延伸误报时 Dice=0，双方都无该类时 Dice 未定义；另列仅参照中存在类别的宏平均。\n',
              '\n## 原始三个模型的100轮曲线\n',
              '曲线来自与原始 U-Net 同批的 V66：U-Net、DeepLabV3+、SegFormer-B0；冻结二值分割与特征后训练时序分类头，使用原始 (18).json。训练和验证损失直接取真实逐轮 history.csv，完整展示1–100轮，无平滑、插值、拼接或重算。原始 epoch0 未训练评估保留在原始CSV中，不算第1轮。选中轮次分别37、14、63，选择依据为验证宏平均Dice，而非最低验证损失。该历史曲线不是最新版人工标注的重新训练，也不是 V90 的训练曲线。\n',
              '\n## 图像\n',
              '图1为四期全部可用上半观察区，含RGB、人工标注、原始U-Net、V90；各分类列同时显示RGB叠加与白底根系。另有四张原生像素单期图。图2为四期同坐标局部，图3–4为六处此前反馈的细节，非根据Dice选取。人工标注列完整保留原标注范围，不伪造标注覆盖。分类图仅用最近邻缩放，TIFF为600dpi、黑色粗体Times New Roman。\n',
              '\n## 下载\n']
    for r in figs:lines.append(f"- [{r['file']}](figures/{r['file']})\n")
    lines += ['- [指标汇总 CSV](Accuracy_and_Dice_summary.csv)\n','- [完整100轮损失 CSV](Three_original_models_actual_100_epoch_losses.csv)\n','- [曲线来源与检查点记录](training_curve_provenance.json)\n']
    (OUT/'RESULTS.md').write_text(''.join(lines))
    # A readable entry page with direct full-size downloads.
    header='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V90 与原始 U-Net 对比</title><style>body{max-width:1400px;margin:30px auto;padding:0 18px;font:16px/1.7 system-ui;color:#18232d}img{width:100%;height:auto;border:1px solid #ddd}section{margin:40px 0}a{color:#075c97}table{border-collapse:collapse}td,th{padding:8px 16px;border:1px solid #ddd}</style><h1>人工标注、原始 U-Net 与 V90 对比</h1><p>人工参照固定为修订后 V87，预测与根系轮廓均未改动。V90 含人工确认单根修正；这里报告同标注一致度，不是独立测试精度。</p><p>四期“全图”为当前所有可用的 2541×1108 上半观察区；T1 为基线，不计入三类时序分数。</p>'''
    body=[header,'<h2>全部有效根系标注区：T2–T4</h2><table><tr><th>版本</th><th>准确率</th><th>持续Dice</th><th>新生Dice</th><th>延伸Dice</th><th>宏平均Dice</th></tr>']
    body.append('<tr><td>人工标注</td><td colspan="5">参照标准；不单独报告模型精度</td></tr>')
    for name,mm in metrics['all_valid_annotated_root_pixels']['T2_T3_T4_combined'].items():body.append(f'<tr><td>{name}</td>'+''.join(f'<td>{v:.4f}</td>' for v in [mm['accuracy']]+[r['Dice'] for r in mm['classes']]+[mm['macro_Dice']])+'</tr>')
    body+=['</table><p><a href="Accuracy_and_Dice_summary.csv">各期及汇总指标CSV</a> · <a href="Per_class_precision_recall_Dice_IoU.csv">分类别完整指标CSV</a> · <a href="Three_original_models_actual_100_epoch_losses.csv">三个模型100轮真实损失CSV</a> · <a href="RESULTS.md">完整说明</a> · <a href="V90_four_period_overview_and_metrics.zip">全图与指标包</a> · <a href="V90_local_comparisons_and_data.zip">局部对比与数据包</a> · <a href="Original_three_models_100_epoch_curves_and_logs.zip">原始训练曲线与日志包</a></p><p>原始三模型曲线为 V66、原 (18).json 的时序训练，非重新训练分割网络，也非 V90 的新训练。真实1–100轮逐点展示。</p>']
    for r in figs:
        stem=Path(r['file']).stem;body.append(f'<section><h2>{stem}</h2><a href="figures/{r["file"]}">下载600dpi TIFF</a>')
        if (figdir/(stem+'.svg')).exists():body.append(f' · <a href="figures/{stem}.svg">矢量SVG</a>')
        body.append(f'<a href="figures/{r["file"]}"><img loading="lazy" src="figures/{stem}_preview.png" alt="{stem}"></a></section>')
    (OUT/'index.html').write_text(''.join(body)+'</html>')
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    js(OUT/'completion.json',dict(state='complete',training_performed=False,predictions_unchanged=True,figures=len(figs),three_model_history_rows=len(histories)))
    finalize_archives()
    print('REPORT_COMPLETE',json.dumps(metrics['all_valid_annotated_root_pixels']['T2_T3_T4_combined']),flush=True)


def finalize_archives():
    """Keep API-uploadable archives small; individual native-resolution TIFFs remain downloadable."""
    warning='逐期退化说明：全部有效标注区 T4 新生根 Dice 为原始 U-Net 0.824105、V90 0.717449，像素准确率为0.988346、0.984794。汇总提高不代表每期、每类都提高；T4 原人工延伸已改新生，而 V90 仍保留部分延伸预测。本报告如实保留该退化，不修改结果。'
    p=OUT/'RESULTS.md';text=p.read_text()
    if warning not in text:p.write_text(text+'\n## 逐期结果的限制\n\n'+warning+'\n')
    p=OUT/'index.html';text=p.read_text()
    if warning not in text:text=text.replace('</html>','<section><h2>逐期结果的限制</h2><p>'+warning+'</p></section></html>')
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    payload=[p for p in sorted(OUT.rglob('*')) if p.is_file() and not any(s.startswith('publication') or s.startswith('published_') for s in p.parts) and p.suffix not in ['.zip','.log'] and p.name not in ['download_links.json','artifact_manifest.json']]
    common=[p for p in payload if p.parent==OUT and p.suffix in ['.csv','.json','.md','.html']]
    groups={
        'V90_four_period_overview_and_metrics.zip':common+[p for p in payload if p.name.startswith(('Fig1_','Fig6_'))],
        'V90_local_comparisons_and_data.zip':common+[p for p in payload if p.name.startswith(('Fig2_','Fig3_','Fig4_')) or 'data' in p.relative_to(OUT).parts or 'source' in p.relative_to(OUT).parts],
        'Original_three_models_100_epoch_curves_and_logs.zip':common+[p for p in payload if p.name.startswith(('Fig5_','Loss_')) or 'histories' in p.relative_to(OUT).parts]
    }
    archives=[]
    for name,files in groups.items():
        archive=OUT/name
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(OUT))
        with zipfile.ZipFile(archive) as z:assert z.testzip() is None
        assert archive.stat().st_size<24_000_000
        archives.append(archive)
    old=OUT/'V90_comparison_and_original_100_epoch_curves.zip'
    if old.exists():old.unlink()
    js(OUT/'artifact_manifest.json',[dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in payload+archives])


if __name__=='__main__': main()
