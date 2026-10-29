"""Two publication figures and two explicitly distinct pooled metric tables."""
import csv, hashlib, json, shutil, sys, zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches

ROOT=Path('/workspace/LI'); HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_V90_SCI_two_figures_20261029'
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
CUR=ROOT/'results/B3_user_confirmed_new_axis_correction_20261029'
REF=ROOT/'results/B3_T4_all_extension_reference_to_new_v87_20261028'
PREVIOUS=ROOT/'results/B3_V90_manual_original_UNet_report_20261029'
BIN=ROOT/'results/B3_annotation_refinement_v1'
PERIODS=['T1','T2','T3','T4']; MODELS={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
ROI=[1530,590,1830,850]  # Previously reported complete new-lateral tip and older-root contacts; no metric ranking.
sys.path.insert(0,str(ROOT/'analysis-scripts/0353-2-B3_segmentation_pilot/root_axis_junction_v83'))
from rendering import isolated,overlay,FONT,PAL


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def workbook(path,sheets):
    """Small standards-based XLSX; numeric values are not formatted strings."""
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    def col(i):
        s=''
        while i:i,k=divmod(i-1,26);s=chr(65+k)+s
        return s
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        ct='<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        for i in range(len(sheets)):ct+=f'<Override PartName="/xl/worksheets/sheet{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        z.writestr('[Content_Types].xml',ct+'</Types>')
        z.writestr('_rels/.rels','<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        wb=f'<?xml version="1.0"?><workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
        rels='<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        for i,(name,rows) in enumerate(sheets,1):
            wb+=f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            rels+=f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            headers=list(rows[0]); data=[headers]+[[r.get(k) for k in headers] for r in rows]
            xml=f'<?xml version="1.0"?><worksheet xmlns="{ns}"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="1" width="31" customWidth="1"/><col min="2" max="{len(headers)}" width="21" customWidth="1"/></cols><sheetData>'
            for ri,row in enumerate(data,1):
                xml+=f'<row r="{ri}">'
                for ci,v in enumerate(row,1):
                    addr=col(ci)+str(ri)
                    if isinstance(v,(int,float)) and not isinstance(v,bool):xml+=f'<c r="{addr}" s="{2 if isinstance(v,float) else 0}"><v>{v}</v></c>'
                    else:xml+=f'<c r="{addr}" t="inlineStr" s="{1 if ri==1 else 0}"><is><t>{escape("" if v is None else str(v))}</t></is></c>'
                xml+='</row>'
            xml+=f'</sheetData><autoFilter ref="A1:{col(len(headers))}{len(data)}"/></worksheet>'
            z.writestr(f'xl/worksheets/sheet{i}.xml',xml)
        z.writestr('xl/workbook.xml',wb+'</sheets></workbook>')
        rels+=f'<Relationship Id="rId{len(sheets)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'
        z.writestr('xl/_rels/workbook.xml.rels',rels)
        z.writestr('xl/styles.xml',f'<?xml version="1.0"?><styleSheet xmlns="{ns}"><numFmts count="1"><numFmt numFmtId="164" formatCode="0.000000"/></numFmts><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills><borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0"/><xf numFmtId="164" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        import xml.etree.ElementTree as ET
        for name in z.namelist():ET.fromstring(z.read(name))


def binary_metrics(c):
    tp,fp,fn,tn=map(int,c)
    return dict(Dice=2*tp/(2*tp+fp+fn),IoU=tp/(tp+fp+fn),Precision=tp/(tp+fp),Recall=tp/(tp+fn),
                Pixel_accuracy=(tp+tn)/(tp+fp+fn+tn),Specificity=tn/(tn+fp),TP=tp,FP=fp,FN=fn,TN=tn,evaluated_pixels=tp+fp+fn+tn)


def class_summary(cm):
    pr=[];rr=[];dd=[];jj=[];tp_total=0
    for k in [1,2,3]:
        tp=int(cm[k,k]);fp=int(cm[:,k].sum())-tp;fn=int(cm[k].sum())-tp
        pr.append(tp/(tp+fp));rr.append(tp/(tp+fn));dd.append(2*tp/(2*tp+fp+fn));jj.append(tp/(tp+fp+fn));tp_total+=tp
    pred_root=int(cm[:,1:].sum());gt_root=int(cm[1:].sum())
    return dict(Macro_Dice=float(np.mean(dd)),Macro_IoU=float(np.mean(jj)),Macro_Precision=float(np.mean(pr)),Macro_Recall=float(np.mean(rr)),
                Root_pixel_accuracy=tp_total/gt_root,Pixel_accuracy_including_background=float(np.trace(cm)/cm.sum()),
                Micro_Dice=2*tp_total/(pred_root+gt_root),Micro_Precision=tp_total/pred_root,Micro_Recall=tp_total/gt_root,
                evaluated_pixels=int(cm.sum()),reference_root_pixels=gt_root)


def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True)
    for name in ['figures','source','histories']:(OUT/name).mkdir(exist_ok=True)
    ref_json=REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json'
    assert sha(ref_json)=='0664587ee2e5931e318f8ad025934b64d5a95fb4ffb8854ac942da861b907636'
    assert sha(BASE/'unet/best_temporal_head.pt')=='58a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382'
    files=[ref_json,BASE/'unet/best_temporal_head.pt',CUR/'verification.json']
    for t in PERIODS:
        files += [BASE/'data/upper'/f'{t}_RGB.png',REF/'corrected_reference'/f'{t}_classes.png',CUR/'full'/f'{t}_classes.png']
        for m in MODELS:files += [BASE/m/'full'/f'{t}_classes.png',BIN/m/'upper'/f'{t}_root_mask.png']
    for m in MODELS:files += [BASE/m/n for n in ['history.csv','config.json','completion.json']]
    frozen={str(p):sha(p) for p in files}
    rgb={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')) for t in PERIODS}
    ref={t:np.array(Image.open(REF/'corrected_reference'/f'{t}_classes.png')) for t in PERIODS}
    pred={m:{t:np.array(Image.open(BASE/m/'full'/f'{t}_classes.png')) for t in PERIODS} for m in MODELS}
    pred['V90']={t:np.array(Image.open(CUR/'full'/f'{t}_classes.png')) for t in PERIODS}
    for t in PERIODS:
        assert np.array_equal(pred['unet'][t]>0,pred['V90'][t]>0)
        for m in MODELS:assert np.array_equal(pred[m][t]>0,np.array(Image.open(BIN/m/'upper'/f'{t}_root_mask.png'))>0)
    assert np.array_equal(pred['unet']['T1'],pred['V90']['T1'])
    doc=json.loads(ref_json.read_text());gt_binary={}
    for e in doc['images']:
        r=e['layers']['mask'];a=np.repeat(np.array(r[::2],np.uint8),np.array(r[1::2],np.int64)).reshape(e['height'],e['width'])[:1108]
        gt_binary[e['name'][:2]]=a
        assert set(np.unique(a)).issubset({0,64,128,255})
    binary=[];classification=[];audit={};period_counts=[]
    for m,pp in pred.items():
        name=MODELS.get(m,'U-Net + V90 (reviewed)');counts=np.zeros(4,np.int64);cm=np.zeros((4,4),np.int64)
        for t in PERIODS:
            a=gt_binary[t];p=pp[t]>0;valid=np.isin(a,[0,255]);gt=a==255
            c=np.array([(gt&p&valid).sum(),(~gt&p&valid).sum(),(gt&~p&valid).sum(),(~gt&~p&valid).sum()],np.int64);counts+=c
            period_counts.append(dict(model=name,period=t,**binary_metrics(c)))
            if t!='T1':
                g=ref[t];v=np.isin(g,[0,1,2,3]);assert np.isin(pp[t][v],[0,1,2,3]).all()
                cm+=np.bincount(g[v].astype(np.int64)*4+pp[t][v],minlength=16).reshape(4,4)
        binary.append(dict(Model=name,**binary_metrics(counts)))
        classification.append(dict(Model=name,**class_summary(cm)))
        audit[name]=dict(binary_counts=counts.tolist(),temporal_confusion_background_persistent_new_extension=cm.tolist())
    assert [r['evaluated_pixels'] for r in binary]==[binary[0]['evaluated_pixels']]*4
    assert [r['evaluated_pixels'] for r in classification]==[classification[0]['evaluated_pixels']]*4
    for k in ['TP','FP','FN','TN']:assert binary[0][k]==binary[-1][k]
    assert binary[0]['FP']==24 and binary[0]['FN']==3
    csvout(OUT/'Table1_Four_period_binary_root_segmentation.csv',binary)
    csvout(OUT/'Table2_Pooled_temporal_class_summary.csv',classification)
    js(OUT/'metric_counts_and_definitions.json',dict(reference_sha256=sha(ref_json),binary_periods=PERIODS,class_summary_periods=PERIODS[1:],
        binary_definition='Pool TP/FP/FN/TN across T1-T4; root=manual mask255 and prediction>0, explicit background=manual mask0; 64/128 ignored. No root-type split.',
        class_definition='Pool the same valid reference pixels across T2-T4, including true background0 and missed roots; macro average foreground classes1/2/3 without listing individual types. T1 baseline is not a change class.',
        root_accuracy_definition='Correct persistent/new/extension predictions divided by all valid manually typed root pixels, including root misses.',
        dependency='The current editable reference foreground almost exactly follows original U-Net; only24 FP and3 FN. Binary near1 is reference-mask agreement, not independent segmentation accuracy. V90 does not change binary masks.',
        reviewed_output='V90 includes explicit annotation-assisted correction; not independent generalization.',counts=audit,per_period_binary_audit=period_counts))
    notes=[dict(Item='Reference',Definition='Revised V87 manual annotation; immutable SHA256 '+sha(ref_json)),
           dict(Item='Binary table',Definition='T1-T4 pooled root/background TP/FP/FN/TN; all explicit valid pixels, not held-out independent images.'),
           dict(Item='Class-summary table',Definition='T2-T4 pooled foreground-class macro metrics; includes binary missed roots and background false positives. T1 is baseline display only.'),
           dict(Item='Near-perfect U-Net binary score',Definition='Manual foreground nearly matches original U-Net:24 FP and3 FN across four periods. This is not independent generalization.'),
           dict(Item='V90',Definition='Same binary geometry as U-Net; includes user-reviewed annotation-assisted single-root correction.'),
           dict(Item='Training curves',Definition='Actual original V66 upload18 temporal-head training, binary/features frozen;100epochs; selected epochs37,14,63.'),
           dict(Item='Local ROI',Definition=str(ROI)+'; same native crop in all four periods; previously reported complete new-lateral tip, not selected by Dice ranking.')]
    workbook(OUT/'Three_models_pooled_metrics.xlsx',[('Binary_T1_T4',binary),('Class_summary_T2_T4',classification),('Definitions',notes)])
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
                         'font.size':9,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    histories={};allrows=[]
    for m in MODELS:
        rows=list(csv.DictReader((BASE/m/'history.csv').open(encoding='utf-8-sig')));rows=[r for r in rows if int(r['epoch'])>=1]
        assert [int(r['epoch']) for r in rows]==list(range(1,101))
        assert np.isfinite([[float(r['training_loss']),float(r['validation_loss'])] for r in rows]).all()
        histories[m]=rows
        for r in rows:allrows.append(dict(Model=MODELS[m],Epoch=int(r['epoch']),Training_loss=float(r['training_loss']),Validation_loss=float(r['validation_loss'])))
        for n in ['history.csv','config.json','completion.json']:shutil.copy2(BASE/m/n,OUT/'histories'/f'{m}_{n}')
    csvout(OUT/'Three_original_models_actual_100_epoch_losses.csv',allrows)
    manifest=[]
    def text(fig,W,H,x,y,label,size=10,ha='left',**kw):
        return fig.text(x/W,y/H,label,fontsize=size,fontweight='bold',ha=ha,va='center',color='black',**kw)
    def image_ax(fig,W,H,a,x,y,w,h,categorical=False,roi=False):
        ih,iw=a.shape[:2];scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale
        ax=fig.add_axes([(x+(w-ww)/2)/W,(y+(h-hh)/2)/H,ww/W,hh/H])
        ax.imshow(a,interpolation='nearest' if categorical else 'antialiased',aspect='equal');ax.set_axis_off()
        if roi:
            x0,y0,x1,y1=ROI
            ax.add_patch(patches.Rectangle((x0-.5,y0-.5),x1-x0,y1-y0,fill=False,edgecolor='#0072B2',linewidth=.85,zorder=4))
        return ax
    def legend(fig,W,H,y):
        start=.65;step=(W-1.3)/4
        for i,(label,k) in enumerate([('T1 baseline',4),('Persistent',1),('Newly observed',2),('Tip extension',3)]):
            x=start+i*step
            fig.add_artist(patches.Rectangle((x/W,(y-.04)/H),.10/W,.10/H,transform=fig.transFigure,facecolor=PAL[k]/255,edgecolor='none'))
            text(fig,W,H,x+.17,y,label,9)
    def save(fig,W,H,stem):
        tif=OUT/'figures'/(stem+'.tif');pdf=OUT/'figures'/(stem+'.pdf')
        fig.savefig(tif,dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(pdf,dpi=300)
        with Image.open(tif) as im:
            assert im.size==(round(W*600),round(H*600));assert np.allclose(im.info['dpi'],[600,600])
            prev=im.convert('RGB');prev.thumbnail((2400,2400));prev.save(OUT/'figures'/(stem+'_preview.png'))
        manifest.append(dict(stem=stem,width_inches=W,height_inches=H,pixels=[round(W*600),round(H*600)],dpi=600,
                             tiff_sha256=sha(tif),pdf_sha256=sha(pdf),PDF='Vector text/curves with 300dpi raster root/RGB panels; lossless TIFF panels remain600dpi'))
        plt.close(fig);print('FIGURE_COMPLETE',stem,tif.stat().st_size,pdf.stat().st_size,flush=True)
    # FIGURE 1: original three-model curves plus full four-period five-column comparison.
    W,H=15.4,15.6;fig=plt.figure(figsize=(W,H),facecolor='white')
    text(fig,W,H,.55,15.20,'a',16);text(fig,W,H,.91,15.20,'Training dynamics',13)
    for i,m in enumerate(MODELS):
        ax=fig.add_axes([(.88+i*5.05)/W,12.18/H,3.95/W,2.55/H])
        rows=histories[m];ep=[int(r['epoch']) for r in rows]
        ax.plot(ep,[float(r['training_loss']) for r in rows],lw=1.1,color='#0072B2',label='Training')
        ax.plot(ep,[float(r['validation_loss']) for r in rows],lw=1.1,color='#D55E00',label='Validation')
        selected=json.loads((BASE/m/'completion.json').read_text())['selected_epoch']
        ax.axvline(selected,color='#727272',ls=(0,(3,3)),lw=.7,label=f'Selected epoch {selected}')
        ax.set_title(MODELS[m],fontsize=12,pad=8);ax.set_xlabel('Epoch',fontsize=9);ax.set_ylabel('Loss',fontsize=9)
        ax.set_xlim(1,100);ax.set_xticks([1,20,40,60,80,100]);ax.tick_params(labelsize=8,length=3,width=.6)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15,linewidth=.6);ax.legend(frameon=False,fontsize=7.5,loc='upper right')
    text(fig,W,H,.55,11.56,'b',16);text(fig,W,H,.91,11.56,'Full observation area across four periods',13)
    xpos=[.67,3.52,6.37,9.22,12.07];panelw=2.66;panelh=panelw*1108/2541
    for x,label in zip(xpos,['Original RGB','Manual reference','U-Net','DeepLabV3+','SegFormer-B0']):text(fig,W,H,x+panelw/2,11.08,label,10.5,ha='center')
    for i,t in enumerate(PERIODS):
        top=10.82-i*2.55;bottom=top-2*panelh-.045
        text(fig,W,H,.29,(top+bottom)/2,t,11.5,ha='center')
        image_ax(fig,W,H,rgb[t],xpos[0],bottom,panelw,2*panelh+.045,roi=True)
        for x,a in zip(xpos[1:],[ref[t]]+[pred[m][t] for m in MODELS]):
            image_ax(fig,W,H,overlay(rgb[t],a),x,top-panelh,panelw,panelh)
            image_ax(fig,W,H,isolated(a),x,bottom,panelw,panelh,True)
    legend(fig,W,H,.32)
    save(fig,W,H,'Fig1_Three_model_100_epoch_curves_and_four_period_full_comparison')
    # FIGURE 2: a fixed, complete root-tip ROI in all periods, three columns only.
    W,H=10.8,12.8;fig=plt.figure(figsize=(W,H),facecolor='white')
    text(fig,W,H,.55,12.48,'Matched local comparison across four periods',13)
    xs=[.65,4.00,7.35];ww=2.70
    for x,label in zip(xs,['Original RGB','Original U-Net','Optimized U-Net (V90)']):text(fig,W,H,x+ww/2,12.05,label,10.5,ha='center')
    x0,y0,x1,y1=ROI;hh=ww*(y1-y0)/(x1-x0)
    for i,t in enumerate(PERIODS):
        top=11.45-i*2.65;y=top-hh;text(fig,W,H,.29,y+hh/2,t,11.5,ha='center')
        for x,a,cat in [(xs[0],rgb[t],False),(xs[1],isolated(pred['unet'][t]),True),(xs[2],isolated(pred['V90'][t]),True)]:
            image_ax(fig,W,H,a[y0:y1,x0:x1],x,y,ww,hh,cat)
    legend(fig,W,H,.53)
    text(fig,W,H,W/2,.15,'Same native ROI: x = 1530-1830, y = 590-850 (300 x 260 pixels)',8,ha='center')
    save(fig,W,H,'Fig2_Four_period_matched_local_RGB_original_UNet_V90')
    js(OUT/'figure_manifest.json',dict(figures=manifest,number_of_scientific_figures=2,ROI=ROI,
        selection='A previously reported complete new-lateral tip and crossings; selected for visible correction and chronology, not ranked by metric gain.',
        plot_style='Bold black Times New Roman, consistent font/line scales, vector curves, native-aspect images, nearest categorical sampling, aligned period rows.',
        curve_source='V66 actual upload18 temporal-head history, not a new V90 training run; epoch0 excluded, epochs1-100 unmodified.',
        full_image_scope='Entire available 2541x1108 registered upper field; lower multi-region canvas is not registered full-tube data.'))
    readme='''# 两张 SCI 综合图与综合指标表

只导出两张科研图，未训练、未修改任何预测、人工标注、根系宽度、长度或轮廓。所有输入逐一SHA256锁定。

**图1**：上方为 U-Net、DeepLabV3+、SegFormer-B0 原始同批 V66 的真实100轮训练/验证损失，使用原 (18).json，二值分割及特征冻结后的时序分类头；选中轮次37/14/63依据验证宏平均Dice，并非最低损失。下方为四期全图，五列为原始RGB、修订人工标注、原始U-Net、DeepLabV3+、SegFormer-B0；分类列同时显示RGB叠加与白底根系。蓝色框标出图2区域。

**图2**：四期同坐标局部，三列仅为原始RGB、原始U-Net、V90优化结果。局部固定为[1530,590,1830,850]，包含完整新生侧根末端与老根交叉；T2原始U-Net绿色根身混入的黄色末端在V90连续为新生。选自此前已反馈的区域，未按Dice排名挑选。其他时期如实展示，未改动颜色。T1为基线；同一根后来转为持续或延伸是正常跨期变化。

两图均为600dpi无损TIFF，并提供PDF。PDF文字与损失曲线为矢量，RGB/分割面板为300dpi栅格，分类图最近邻缩放；600dpi图像细节以TIFF为准。全图为当前完整可用的2541×1108上半观察区，不代表整根管。

## 两种指标口径

1. `Table1_Four_period_binary_root_segmentation.csv`：把全部时序类型合并为根系前景，四期先合并TP/FP/FN/TN，再计算Dice、IoU、Precision、Recall、Accuracy、Specificity。人工mask255为根、0为明确背景，64/128忽略。并非先算每期Dice再简单平均。
2. `Table2_Pooled_temporal_class_summary.csv`：不逐类展开，将T2–T4有效像素先合并，再对持续/新生/延伸三个前景类别计算宏平均Dice、IoU、Precision、Recall；同时提供微平均与根像素准确率。包含分割漏检和背景误检；未限定在各模型各自预测根系内。T1仅基线，不是变化类别，故不硬凑进三类时序宏平均。

人工参照固定为修订V87，含此前获授权的T2两处与T4延伸改新生；不为本次比较再次改标注。该参照的根系轮廓几乎直接沿用原始U-Net分割，四期只有24个背景误检、3个根漏检，所以U-Net根系二值Dice接近1；这是对该人工轮廓的一致度，不能当作独立分割泛化精度。V90冻结二值前景，其二值指标必须与原始U-Net完全相同。

V90包括明确人工复核单根修正，且整个位置已用于反馈开发，所有指标均为同位置开发标注一致度，不是独立测试准确率。V90未优化二值分割；其影响体现在类别综合与局部连接。T4新生单类指标存在已报告退化，此处按你的要求不逐类展示；不能由综合提升推断每期、每类均提升。

Excel含两张综合指标表与Definitions说明页。CSV保留全精度，Excel显示六位小数。`metric_counts_and_definitions.json`保留原始计数、混淆矩阵和口径，支持复算。三个历史训练日志均逐字节保留。
'''
    def mdtable(rows,keys):
        return '| '+' | '.join(keys)+' |\n|'+ '|'.join(['---']*len(keys))+'|\n'+''.join('| '+' | '.join(f'{r[k]:.6f}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |\n' for r in rows)
    readme+='\n### 四期二值根系汇总\n\n'+mdtable(binary,['Model','Dice','IoU','Precision','Recall','Pixel_accuracy'])
    readme+='\n### 类别综合汇总（不逐类）\n\n'+mdtable(classification,['Model','Macro_Dice','Macro_IoU','Macro_Precision','Macro_Recall','Root_pixel_accuracy'])
    (OUT/'RESULTS.md').write_text(readme)
    (OUT/'Figure_captions.txt').write_text('Figure 1. Original three-model temporal training and four-period full-field comparison. (a) Actual training and validation losses over100epochs, with validation-macro-Dice-selected epochs marked. Original training used upload18 with frozen binary segmentation/features. (b) Registered RGB, revised manual reference and original U-Net, DeepLabV3+ and SegFormer-B0 temporal maps; overlay and isolated-root views are paired. Blue rectangles identify the matched region in Figure2.\n\nFigure 2. Matched local RGB/original-U-Net/V90 comparison over four periods at native ROI[1530,590,1830,850]. The previously reported new lateral shows an erroneously yellow tip in the original T2 map and continuous green identity after correction. T1 is baseline. V90 includes annotation-assisted review; figures are development examples, not independent test evidence. Root silhouettes and input images are unchanged.\n')
    for p,h in frozen.items():assert sha(p)==h
    js(OUT/'verification.json',dict(passed=True,immutable_input_sha256=frozen,no_training=True,no_predictions_or_annotations_modified=True,
        exactly_two_figures=True,all_three_loss_histories_complete_1_to_100=True,full_native_field_and_one_fixed_local_ROI=True,
        identical_binary_and_classification_supports_across_models=True,V90_binary_equals_original_UNet=True,
        foreground_reference_dependency_explicit=True,metric_class_summary_includes_missed_roots_and_false_positives=True,
        xlsx_XML_and_ZIP_integrity_passed=True,not_independent_test=True))
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    js(OUT/'completion.json',dict(state='complete',scientific_figures=2,metric_scopes=2,epochs=100,training_performed=False))
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','download_links.json'] and not p.name.startswith(('published_','publication')) and p.suffix not in ['.zip','.log']]
    js(OUT/'artifact_manifest.json',[dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files])
    print('COMPLETE',json.dumps(dict(binary=binary,classification=classification)),flush=True)


if __name__=='__main__':main()
