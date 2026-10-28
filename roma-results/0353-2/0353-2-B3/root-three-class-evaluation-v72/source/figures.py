"""Show class-specific, within-date and across-date identity results separately."""
import csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
from evaluate import ROOT,OUT,MODELS,NAMES,CLASSES,sha,js,csvout
def read(p):
    with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    rows=read(OUT/'Three_classes_connection_by_period.csv');allrows=read(OUT/'Three_classes_connection_metrics.csv');outputs=[]
    font=ROOT/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'
    for _,cls,chinese in CLASSES:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
        for ax,scope,title in zip(axes,['within_period','across_adjacent_periods'],['Within-period segment connection','Adjacent-period continuation']):
            for i,(variant,label,color) in enumerate([('mlp','No-GAT identity control','#0072B2'),('gat','Identity GAT','#D55E00')]):
                vals=[float(next(r['connection_F1'] for r in rows if r['model']==m and r['variant']==variant and r['category']==cls and r['pair_scope']==scope)) for m in MODELS]
                bars=ax.bar(np.arange(3)+(i-.5)*.32,vals,.30,label=label,color=color);ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
            labels=[]
            for m in MODELS:
                r=next(r for r in rows if r['model']==m and r['variant']=='gat' and r['category']==cls and r['pair_scope']==scope)
                labels.append(NAMES[m]+'\npos='+r['positive_pairs']+', neg='+r['negative_pairs'])
            ax.set_xticks(np.arange(3),labels);ax.set_ylim(0,1.18);ax.set_ylabel('Same-root connection F1');ax.set_title(title);ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8,loc='upper left')
        fig.suptitle(cls.replace('_',' ').title()+' — stratified by reference category, not pixel classification',fontsize=12)
        fig.canvas.draw()
        for tx in fig.findobj(Text):tx.set_color('black');tx.set_fontproperties(FontProperties(fname=str(font),size=tx.get_fontsize(),weight='bold'))
        stem='Fig_'+cls+'_connection_evaluation';fig.savefig(OUT/(stem+'_preview.png'),dpi=160,facecolor='white');tmp=OUT/(stem+'_full.png');fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
        path=OUT/(stem+'.tif')
        with Image.open(tmp) as im:im.convert('RGB').save(path,compression='tiff_adobe_deflate',dpi=(600,600))
        tmp.unlink();outputs.append(dict(file=path.name,sha256=sha(path)))
    summary=[]
    for m in MODELS:
        rec=dict(model=NAMES[m])
        for _,cls,chinese in CLASSES:
            for scope in ['within_period','across_adjacent_periods']:
                r=next(r for r in rows if r['model']==m and r['variant']=='gat' and r['category']==cls and r['pair_scope']==scope);rec[cls+'_'+scope+'_F1']=r['connection_F1'];rec[cls+'_'+scope+'_positive_pairs']=r['positive_pairs']
        summary.append(rec)
    csvout(OUT/'GAT_three_classes_summary.csv',summary)
    lines=['# 新生、延伸、持续分别评估','',
        'V71的三组GAT及三组无GAT对照均已完成100轮。本轮只学习根段归属与连接，没有重训时序像素分类。',
        '本报告按(18)参考标注的类别对已冻结预测分层，连接正负标签来自(17)的142个已确认编号。这里的F1衡量根段能否接对，不是三类像素分类Dice，也不表示模型已经识别了这些类别。',
        '类别锚点至少含3个留出标注像素、且90%以上同类；同根的持续段与延伸段之间、或新生到后期持续之间的正确连接仍保留为正例。每条边在每个类别内只计一次；一条边可属于两个类别，类别样本量不能相加。',
        '所有类别与时间组都沿用每个分支原先由验证集确定的同一个阈值，没有按测试类别调参。样本来自单一B3视野的内部空间留出，不是独立位置或根系泛化。','',
        '## 同一期根段连接F1','',
        '| 模型 | 新生 | 延伸 | 持续 |','|---|---:|---:|---:|']
    for m in MODELS:
        vals=[float(next(r['connection_F1'] for r in rows if r['model']==m and r['variant']=='gat' and r['category']==cls and r['pair_scope']=='within_period')) for _,cls,_ in CLASSES]
        lines.append('| '+NAMES[m]+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
    lines+=['','## 跨相邻期延续F1','','| 模型 | 新生标注根段相关 | 延伸标注根段相关 | 持续标注根段相关 |','|---|---:|---:|---:|']
    for m in MODELS:
        vals=[float(next(r['connection_F1'] for r in rows if r['model']==m and r['variant']=='gat' and r['category']==cls and r['pair_scope']=='across_adjacent_periods')) for _,cls,_ in CLASSES]
        lines.append('| '+NAMES[m]+' | '+' | '.join(f'{x:.4f}' for x in vals)+' |')
    lines+=['',
        '关键发现：合并指标主要受较容易的跨期连接影响，不能代表同一期交叉/断段附近的连接能力。分层后，同一期新生与持续根连接明显更弱，正是下一步应重点改进的部分。',
        '延伸段同一期正连接仅U-Net9对、DeepLabV3+3对、SegFormer12对；DeepLab的1.0000仅来自3个正例，不足以证明已解决延伸或交叉识别。',
        'Three_classes_original_temporal_baseline.csv另存原V66冻结时序分类的Precision、Recall、Dice、IoU。它是原模型整幅推理基线，未受V71归属训练改变，不能称作新GAT的分类结果。',
        '原始标注、根号、模型权重与阈值、根系轮廓、宽度和长度均未修改；所有缺失/混合/样本不足情况保留在核对记录中。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n');js(OUT/'figure_manifest.json',dict(files=outputs,dpi=600,font='Times New Roman Bold, black',no_smoothing=True))
    print('Saved3 class figures and stratified result tables',flush=True)
if __name__=='__main__':main()
