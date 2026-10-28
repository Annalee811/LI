"""Publish real class-specific outcomes, including declines and the MLP control."""
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from prepare import ROOT,OUT,BASE,MODELS,PERIODS,rows,js,sha

FONT=ROOT/'results/B3_fixed_100_epochs_v35_20261023/assets/TimesNewRoman-Bold.ttf'
NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
PALETTE=np.array([[255,255,255],[136,86,167],[0,158,115],[230,171,2],[0,114,178]],np.uint8)
def finish(fig,stem):
    fig.canvas.draw()
    for t in fig.findobj(Text):t.set_color('black');t.set_fontproperties(FontProperties(fname=str(FONT),size=t.get_fontsize(),weight='bold'))
    png=OUT/(stem+'_preview.png');fig.savefig(png,dpi=160,facecolor='white')
    tmp=OUT/(stem+'_full.png');fig.savefig(tmp,dpi=600,facecolor='white');plt.close(fig)
    tif=OUT/(stem+'.tif')
    with Image.open(tmp) as im:im.convert('RGB').save(tif,compression='tiff_adobe_deflate',dpi=(600,600))
    tmp.unlink();return dict(file=tif.name,sha256=sha(tif))
def main():
    metrics=rows(OUT/'Temporal_class_metrics_before_after.csv');connection=rows(OUT/'Connection_metrics_GAT_control.csv');continuity=rows(OUT/'Class_continuity_comparison.csv');files=[]
    for cls,title,stem in [('persistent','Persistent roots','Persistent'),('newly_observed','Newly observed roots','New_roots'),('tip_extension','Tip extension','Tip_extension')]:
        fig,axes=plt.subplots(1,2,figsize=(10,3.9),layout='constrained')
        for ax,field in zip(axes,['Dice','IoU']):
            for i,(variant,label,color) in enumerate([('baseline','Original temporal branch','#B5BDC6'),('mlp','Focused training, no GAT','#0072B2'),('gat','Focused training + GAT','#D55E00')]):
                vals=[float(next(r[field] for r in metrics if r['model']==m and r['variant']==variant and r['name']==cls)) for m in MODELS]
                bars=ax.bar(np.arange(3)+(i-1)*.24,vals,.22,label=label,color=color)
                ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
            ax.set_xticks(np.arange(3),[NAMES[m] for m in MODELS]);ax.set_ylabel(field);ax.set_ylim(0,1.12);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        axes[0].legend(frameon=False,fontsize=8,loc='upper left');fig.suptitle(title+' — classification within fixed root masks')
        files.append(finish(fig,'Fig_'+stem+'_before_after_GAT'))
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,m in zip(axes,MODELS):
        for kind,label,col in [('mlp','Focused no-GAT control','#0072B2'),('gat','Focused GAT','#D55E00')]:
            h=rows(OUT/(m+'_'+kind)/'history.csv');ax.plot([int(v['epoch']) for v in h],[float(v['training_loss']) for v in h],label=label,color=col,lw=1.5)
        ax.set_title(NAMES[m]);ax.set_xlabel('Refinement training epoch');ax.set_ylabel('Training objective');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    files.append(finish(fig,'Fig_100_epoch_GAT_control_training'))
    fig,axes=plt.subplots(1,3,figsize=(13.5,3.8),layout='constrained')
    for i,(kind,label,col) in enumerate([('mlp','Focused no-GAT control','#0072B2'),('gat','Focused GAT','#D55E00')]):
        vals=[float(next(r['F1'] for r in connection if r['model']==m and r['variant']==kind)) for m in MODELS]
        axes[0].bar(np.arange(3)+(i-.5)*.32,vals,.30,label=label,color=col)
    for i,(variant,label,col) in enumerate([('baseline','Original branch','#B5BDC6'),('mlp','Focused no-GAT','#0072B2'),('gat','Focused GAT','#D55E00')]):
        vals=[]
        for m in MODELS:
            s=next(r['unnecessary_class_switch_rate'] for r in continuity if r['model']==m and r['variant']==variant);vals.append(float(s) if s else np.nan)
        axes[1].bar(np.arange(3)+(i-1)*.24,vals,.22,label=label,color=col)
        boundary=[]
        for m in MODELS:
            s=next(r['true_transition_preservation_rate'] for r in continuity if r['model']==m and r['variant']==variant);boundary.append(float(s) if s else np.nan)
        axes[2].bar(np.arange(3)+(i-1)*.24,boundary,.22,label=label,color=col)
    for ax in axes:ax.set_xticks(np.arange(3),[NAMES[m] for m in MODELS]);ax.set_ylim(0,1);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    from matplotlib.ticker import PercentFormatter
    axes[1].set_ylim(0,.08);axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Same-root candidate connection F1');axes[1].set_ylabel('Unnecessary class-switch fraction');axes[1].set_title('Only reference-constant same-root pairs')
    axes[2].set_ylabel('True class-transition preservation');axes[2].set_title('Persistent-to-extension boundaries retained')
    counts=[int(next(r['reference_true_class_transition_pairs'] for r in continuity if r['model']==m and r['variant']=='baseline')) for m in MODELS]
    axes[2].set_xticklabels([NAMES[m]+'\n(n='+str(n)+')' for m,n in zip(MODELS,counts)])
    axes[2].text(2,.08,'Not estimable' if counts[2]==0 else '',ha='center',fontsize=8)
    files.append(finish(fig,'Fig_Connection_and_class_continuity'))
    # Four-period baseline/GAT geometry comparison, direct from native class maps.
    def text(draw,x,y,label,size=44):draw.text((x,y),label,font=ImageFont.truetype(str(FONT),size),fill='black',anchor='mt')
    def isolated(a,reference_mask=None):
        out=np.full((*a.shape,3),255,np.uint8);valid=np.isin(a,[1,2,3,4]);out[valid]=PALETTE[a[valid]]
        if reference_mask is not None:out[reference_mask&~valid]=[148,148,148]
        return out
    for m in MODELS:
        image=Image.new('RGB',(4900,4200),'white');draw=ImageDraw.Draw(image)
        text(draw,2450,35,NAMES[m]+': temporal classification before and after GAT',76)
        text(draw,2450,140,'100 refinement epochs | Fixed binary contours, widths and lengths | Reference: upload18',36)
        xs=[170,1350,2550,3750];w=1000;h=436
        for x,label in zip(xs,['Original RGB','Reference','Original temporal branch','Focused training + GAT']):text(draw,x+w//2,240,label,42)
        for i,t in enumerate(PERIODS):
            y=310+i*925;text(draw,75,y+440,t,60)
            rgb=np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png'));raw=Image.fromarray(rgb).resize((w,h),Image.Resampling.LANCZOS);image.paste(raw,(xs[0],y+220))
            for x,variant in zip(xs[1:],['reference','baseline','gat']):
                if variant=='reference':
                    a=np.array(Image.open(BASE/'data/current'/f'{t}_classes.png'));mask=np.array(Image.open(BASE/'data/full_annotation'/f'{t}_mask.png'))[:1108]==255
                else:a=np.array(Image.open(OUT/(m+'_'+variant)/'full'/f'{t}_classes.png'));mask=None
                color=isolated(a,mask);valid=np.isin(a,[1,2,3,4]);over=rgb.copy();over[valid]=np.rint(.25*rgb[valid]+.75*PALETTE[a[valid]]).astype(np.uint8)
                image.paste(Image.fromarray(over).resize((w,h),Image.Resampling.LANCZOS),(x,y));image.paste(Image.fromarray(color).resize((w,h),Image.Resampling.NEAREST),(x,y+h+15))
            draw.line((150,y+900,4800,y+900),fill='#E0E5EB',width=2)
        for j,(label,k) in enumerate([('T1 baseline',4),('Persistent',1),('Newly observed',2),('Tip extension',3)]):
            x=230+j*1190;draw.rectangle((x,4090,x+36,4126),fill=tuple(PALETTE[k]));text(draw,x+240,4088,label,36)
        path=OUT/f'Fig_{m}_four_period_before_after_GAT.tif';image.save(path,compression='tiff_adobe_deflate',dpi=(600,600));image.thumbnail((1800,1700));image.save(OUT/f'Fig_{m}_four_period_before_after_GAT_preview.png');files.append(dict(file=path.name,sha256=sha(path)))
    js(OUT/'figure_manifest.json',dict(files=files,font='Times New Roman Bold, all text black',dpi=600,smoothing=False))
    # Rewrite the outcome report around the actual heldout results, including unfavorable changes.
    lines=['# 新生根与根尖延伸：GAT优化结果','',
        '本次时序标签只使用(18)，连接监督只使用用户授权的(17)中142个user_defined根编号；自动提议和未确认编号不参与监督。图输入不含编号、标注颜色或母根信息。',
        '原有二值根系轮廓、宽度和长度完全固定。新GAT从随机初始化开始，使用本次V66的冻结时序概率；不继承旧GAT参数。三组新增GAT及三组同输入MLP对照各100轮真实训练。',
        '评价为同一B3位置的内部空间留出一致性，四期同属一个开发位置，非独立根/管/位置泛化。主指标限定在已固定分割根系内；另存包含二值漏检的端到端指标。',
        '本次原分支和优化分支统一按整幅图滑窗融合后的概率，在完全相同的留出像素上重新计分；之前V66报告按单个留出分块直接预测累积，因此基线数值可能略有差异。不能把两种推理口径的差异当作GAT提升。','',
        '| 模型 | 类别 | 原分支 Dice | 加强训练无GAT | 加强训练+GAT |','|---|---|---:|---:|---:|']
    for m in MODELS:
        for cls,chinese in [('persistent','持续根'),('newly_observed','新生根'),('tip_extension','根尖延伸')]:
            vals=[float(next(r['Dice'] for r in metrics if r['model']==m and r['variant']==v and r['name']==cls)) for v in ['baseline','mlp','gat']]
            lines.append('| '+NAMES[m]+' | '+chinese+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','连接F1仅针对几何候选边、且两端均可由已确认编号判定的内部留出样本；未经确认的交叉不作为负样本。类别相同不等于同根。',
        '中途换类率只在人工编号相同、参考类别本应相同的根段对上计算。真实持续—延伸类别边界另外保存保留率，不能靠把整条根涂成同一色来声称改善。',
        '真实类别过渡边界的可评分根段对仅U-Net5对、DeepLabV3+2对、SegFormer0对，因此这项保留率样本不足，不能据此声称普遍改善根尖边界。',
        '本轮没有稳定改善目标：三个模型新生根Dice均略降，根尖延伸仅后两个模型小幅提高。U-Net与DeepLab的不合理换类率高于原分支；GAT连接F1均低于同输入无GAT对照。建议保留原模型作为当前使用版本，本轮GAT仅作为完整保留的实验结果。',
        '全部训练、验证选择、最终评估及不利变化均保留。测试结果不用于调参重跑或筛选展示窗口。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
