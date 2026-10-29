"""Add the unchanged manual reference to Figure2; preserve prior data and figures."""
import json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_V90_SCI_Fig2_manual_reference_20261029'
sys.path.insert(0,str(HERE.parent/'v90_sci_two_figures'))
import export as old
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib import patches


def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
    files=[old.REF/'root_annotation_progress_24_user_corrected_T2_and_T4.json',old.OUT/'Three_models_pooled_metrics.xlsx']
    files += list((old.OUT/'figures').glob('Fig1_*'))
    for t in old.PERIODS:
        files += [old.BASE/'data/upper'/f'{t}_RGB.png',old.REF/'corrected_reference'/f'{t}_classes.png',old.BASE/'unet/full'/f'{t}_classes.png',old.CUR/'full'/f'{t}_classes.png']
    hashes={str(p):old.sha(p) for p in files}
    assert old.sha(files[0])=='0664587ee2e5931e318f8ad025934b64d5a95fb4ffb8854ac942da861b907636'
    font_manager.fontManager.addfont(str(old.FONT))
    plt.rcParams.update({'font.family':'Times New Roman','font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
                         'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    W,H=14.15,12.8;fig=plt.figure(figsize=(W,H),facecolor='white')
    def text(x,y,label,size=10,ha='left'):
        fig.text(x/W,y/H,label,fontsize=size,fontweight='bold',ha=ha,va='center',color='black')
    text(.55,12.48,'Matched local comparison across four periods',13)
    xs=[.65,4.00,7.35,10.70];ww=2.70
    columns=['Original RGB','Manual annotation','Original U-Net','Optimized U-Net (V90)']
    for x,label in zip(xs,columns):text(x+ww/2,12.05,label,10.5,'center')
    x0,y0,x1,y1=old.ROI;hh=ww*(y1-y0)/(x1-x0);panels=[]
    for i,t in enumerate(old.PERIODS):
        sources=[old.BASE/'data/upper'/f'{t}_RGB.png',old.REF/'corrected_reference'/f'{t}_classes.png',old.BASE/'unet/full'/f'{t}_classes.png',old.CUR/'full'/f'{t}_classes.png']
        arrays=[np.array(Image.open(p)) for p in sources]
        assert all(a.shape[:2]==(1108,2541) for a in arrays)
        top=11.45-i*2.65;y=top-hh;text(.29,y+hh/2,t,11.5,'center')
        for j,(x,a) in enumerate(zip(xs,arrays)):
            crop=a[y0:y1,x0:x1]
            im=crop if j==0 else old.isolated(crop)
            ax=fig.add_axes([x/W,y/H,ww/W,hh/H])
            ax.imshow(im,interpolation='antialiased' if j==0 else 'nearest',aspect='equal');ax.set_axis_off()
            panels.append(dict(period=t,column=columns[j],source=str(sources[j]),source_sha256=hashes[str(sources[j])],native_ROI=old.ROI,
                               native_crop_shape=list(crop.shape),manual_ignore_pixels=int((crop==255).sum()) if j==1 else None))
    start=.65;step=(W-1.3)/4
    for i,(label,k) in enumerate([('T1 baseline',4),('Persistent',1),('Newly observed',2),('Tip extension',3)]):
        x=start+i*step
        fig.add_artist(patches.Rectangle((x/W,(.53-.04)/H),.10/W,.10/H,transform=fig.transFigure,facecolor=old.PAL[k]/255,edgecolor='none'))
        text(x+.17,.53,label,9)
    text(W/2,.15,'Same native ROI: x = 1530-1830, y = 590-850 (300 x 260 pixels)',8,'center')
    stem='Fig2_Four_period_matched_local_RGB_manual_original_UNet_V90'
    tif=OUT/'figures'/(stem+'.tif');pdf=OUT/'figures'/(stem+'.pdf')
    fig.savefig(tif,dpi=600,pil_kwargs={'compression':'tiff_adobe_deflate'});fig.savefig(pdf,dpi=300);plt.close(fig)
    with Image.open(tif) as im:
        assert im.size==(8490,7680);assert np.allclose(im.info['dpi'],[600,600])
        preview=im.convert('RGB');preview.thumbnail((2400,2400));preview.save(OUT/'figures'/(stem+'_preview.png'))
    for p,h in hashes.items():assert old.sha(p)==h
    old.js(OUT/'figure_manifest.json',dict(figure=2,columns=columns,periods=old.PERIODS,ROI=old.ROI,panels=panels,
        tiff_dpi=600,tiff_pixels=[8490,7680],tiff_sha256=old.sha(tif),pdf_sha256=old.sha(pdf),
        font='Black bold embedded Times New Roman',category_resampling='nearest',manual_source='Unchanged revised V87 reference',
        reference_not_inferred_or_filled=True,previous_figure1_and_metric_tables_unchanged=True))
    old.js(OUT/'verification.json',dict(passed=True,immutable_input_sha256=hashes,
        manual_annotation_column_added=True,all_four_columns_same_native_ROI=True,model_predictions_unchanged=True,
        manual_reference_unchanged=True,previous_figure1_unchanged=True,previous_metrics_unchanged=True,
        no_training=True,no_evaluation_recalculation=True))
    (OUT/'RESULTS.md').write_text('''# 图2加入人工标注对照

四期四列：原始RGB、人工标注、原始U-Net、V90。局部坐标沿用[1530,590,1830,850]，各列同尺寸、同比例，原有RGB和预测数据、人工标注均未修改，也未重训或重新计算指标。图1及上一版指标表保持不变。

人工列读取与图1及指标表相同的修订V87人工参照，保留此前授权的T2两处和T4延伸改新生；没有将V90输出冒充人工标注，没有补填人工未标处。T1为基线。分类图使用最近邻缩放；600dpi无损TIFF为8490×7680像素，PDF文字矢量、图像面板300dpi。

V90含人工复核修正，此图为同位置开发对照，不是独立测试证据。各期变化完整呈现，不能据局部示例推断所有错误已消除。
''')
    (OUT/'Figure2_caption.txt').write_text('Figure 2. Matched four-period local comparison of registered RGB, revised manual annotation, original U-Net and optimized V90. All four columns use the same native ROI[1530,590,1830,850]. The manual column is the unchanged V87 reference; no predictions or labels were modified. T1 is baseline. V90 includes annotation-assisted user review; the comparison is a development example, not independent test evidence.\n')
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    old.js(OUT/'completion.json',dict(state='complete',figure2_columns=4,figure1_and_metrics_unchanged=True))
    paths=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','download_links.json'] and not p.name.startswith(('publication','published_')) and p.suffix!='.log']
    old.js(OUT/'artifact_manifest.json',[dict(file=str(p.relative_to(OUT)),sha256=old.sha(p),bytes=p.stat().st_size) for p in paths])
    print('COMPLETE',tif.stat().st_size,pdf.stat().st_size,flush=True)


if __name__=='__main__':main()
