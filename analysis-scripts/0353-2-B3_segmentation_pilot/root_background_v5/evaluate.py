#!/usr/bin/env python3
"""Freeze validation ranking, then produce strict root-only metrics and figures."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as c

NAMES={'unet':'U-Net','deeplab':'DeepLabV3+','segformer':'SegFormer-B0'}
def read(path):return json.loads(Path(path).read_text())
def checkpoint(root,name):return root/name/('best_weights_fp16.npz' if name=='deeplab' else 'best.pt')
def select(root):
    candidates=[]
    for name in NAMES:
        run=read(root/name/'training_run.json');assert c.sha(checkpoint(root,name))==run['checkpoint_sha256']
        m=read(root/name/'validation_metrics.json')['root']
        candidates.append(dict(model=name,validation=m,frozen_threshold=run['frozen_threshold'],checkpoint_sha256=run['checkpoint_sha256']))
    winner=max(candidates,key=lambda x:(x['validation']['Dice'],x['validation']['Recall']))['model']
    record=dict(winner=winner,criterion='Validation strict pixel root Dice only, Recall breaks exact ties',
        test_used_to_select=False,candidates=candidates,task='binary root/background only')
    if (root/'selection.json').exists():assert read(root/'selection.json')==record,'Selection already frozen and changed'
    c.js(root/'selection.json',record);print('ROOT_WINNER',winner,flush=True)

def length_field(mask):
    return edge_length_field(skeletonize(mask))
def edge_length_field(sk):
    out=np.zeros(sk.shape,np.float64)
    for dy,dx in [(0,1),(1,0),(1,1),(1,-1)]:
        y1=slice(0,-dy) if dy else slice(None);y2=slice(dy,None)
        x1=slice(0,-dx) if dx>0 else (slice(-dx,None) if dx<0 else slice(None))
        x2=slice(dx,None) if dx>0 else (slice(0,dx) if dx<0 else slice(None))
        edges=sk[y1,x1]&sk[y2,x2]
        if dx and dy:edges&=~(sk[y1,x2]|sk[y2,x1])
        edge=edges*.5*math.hypot(dy,dx);out[y1,x1]+=edge;out[y2,x2]+=edge
    isolated=sk&(ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')==1)
    out[isolated]=1;return out
def agreement(x,y):
    x=np.array(x,float);y=np.array(y,float);d=y-x;sst=((x-x.mean())**2).sum();sd=d.std(ddof=1)
    cov=((x-x.mean())*(y-y.mean())).mean();den=x.var()+y.var()+(x.mean()-y.mean())**2
    return dict(n_descriptive_pairs=len(x),R2_agreement=float(1-(d*d).sum()/sst) if sst>0 else None,
        RMSE_px=float(np.sqrt((d*d).mean())),MAE_px=float(abs(d).mean()),mean_bias_px=float(d.mean()),
        CCC=float(2*cov/den) if den else None,Pearson_r=float(np.corrcoef(x,y)[0,1]) if x.var()>0 and y.var()>0 else None,
        Bland_Altman_lower_px=float(d.mean()-1.96*sd),Bland_Altman_upper_px=float(d.mean()+1.96*sd),
        reference='Human mask-derived skeleton length, not separately measured manual length',units='native pixels',
        dependence='8 nonoverlapping blocks repeated across4 periods; not32 independent biological replicates')
def savefig(fig,path):
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=300);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)

def report(root):
    data=c.Data(root);selection=read(root/'selection.json');allmetrics={};metricrows=[];lengthrows=[];lengthmetrics={};topology={};checks=[]
    provenance=read(root/'dataset/provenance.json')
    assert c.sha(root/'original_human_progress.json')==provenance['source_progress_sha256']
    for path,expected in provenance['files'].items():assert c.sha(root/path)==expected,'Training input changed: '+path
    figures=root/'figures';figures.mkdir(exist_ok=True)
    for name in NAMES:
        run=read(root/name/'training_run.json');threshold=run['frozen_threshold']
        assert c.sha(checkpoint(root,name))==run['checkpoint_sha256']==next(q['checkpoint_sha256'] for q in selection['candidates'] if q['model']==name)
        allmetrics[name]={};lengthmetrics[name]={};topology[name]={}
        for split,cols in [('train',[0,1,2]),('val',[3]),('test',[4])]:
            total=np.zeros(4,np.int64);per={};ls=[];sgtotal=sptotal=sginside=spinside=0
            for col in cols:
                for t in c.PERIODS:
                    folder=root/name/'predictions'/split/('C'+str(col));pr=np.load(folder/(t+'_probability.npy'))
                    assert pr.shape==(1108,448) and pr.dtype==np.float32 and np.isfinite(pr).all() and ((pr>=0)&(pr<=1)).all()
                    gt=data.gt[t][:,c.COLS[col]:c.COLS[col]+448];pred=pr>=threshold
                    cc=c.counts(gt,pred);total+=cc;per[t]=per.get(t,np.zeros(4,np.int64))+cc
                    Image.fromarray(pred.astype(np.uint8)*255).save(folder/(t+'_root_mask.png'))
                    Image.fromarray(gt).save(folder/(t+'_human_mask.png'))
                    if split!='train':
                        sg=skeletonize(gt==255);sp=skeletonize(pred);sgtotal+=int(sg.sum());sptotal+=int(sp.sum())
                        sginside+=int((sg&pred).sum());spinside+=int((sp&(gt==255)).sum())
                        lf=length_field(gt==255);lp=length_field(pred)
                        for yi in range(4):
                            for xi in range(2):
                                yy=slice(yi*277,(yi+1)*277);xx=slice(xi*224,(xi+1)*224)
                                row=dict(model=name,split=split,period=t,block='X'+str(xi)+'Y'+str(yi),
                                    human_label_length_px=float(lf[yy,xx].sum()),predicted_length_px=float(lp[yy,xx].sum()))
                                ls.append(row);lengthrows.append(row)
            m=c.metrics(total);allmetrics[name][split]=dict(root=m,per_period={t:c.metrics(v) for t,v in per.items()},threshold=threshold)
            stored=read(root/name/('validation_metrics.json' if split=='val' else split+'_metrics.json'))['root']
            assert m==stored,'Saved maps must reproduce reported scores exactly'
            checks.append(dict(model=name,split=split,maps_reproduce_metrics=True))
            for t,mm in [('all',m),*[(t,c.metrics(v)) for t,v in per.items()]]:
                metricrows.append(dict(model=name,split=split,period=t,threshold=threshold,**mm))
            if split!='train':
                lengthmetrics[name][split]=agreement([r['human_label_length_px'] for r in ls],[r['predicted_length_px'] for r in ls])
                tp=spinside/max(1,sptotal);ts=sginside/max(1,sgtotal)
                topology[name][split]=dict(skeleton_precision=tp,skeleton_recall=ts,clDice=2*tp*ts/max(1e-12,tp+ts),
                    interpretation='Separate topology metric; no dilation or tolerance; strict pixel Dice remains the gate')
    c.js(root/'all_metrics.json',allmetrics);c.csvout(root/'segmentation_metrics.csv',metricrows)
    c.csvout(root/'root_length_pairs.csv',lengthrows);c.js(root/'root_length_agreement.json',lengthmetrics);c.js(root/'topology_metrics.json',topology)
    c.js(root/'evaluation_verification.json',dict(passed=True,checks=checks,weights_match_frozen_selection=True,
        same_native_pixels_counted_once=True,source_and_all_input_hashes_still_match=True,
        inference_inputs='Current RGB only; manual labels read only for training or scoring, never inference'))
    winner=selection['winner'];fail=[]
    for split in ['val','test']:
        m=allmetrics[winner][split]['root']
        for key,bound in [('Dice',.95),('Precision',.90),('Recall',.95)]:
            if m[key]<bound:fail.append(split+' '+key+' <'+str(bound))
    gate=dict(root_model=winner,root_checkpoint_sha256=c.sha(checkpoint(root,winner)),
        root_threshold=read(root/winner/'training_run.json')['frozen_threshold'],
        enabled_to_start_temporal_stage=not fail,temporal_stage_started=False,other_position_prediction_performed=False,
        criterion='Validation AND same-position spatial holdout: root Dice>=0.95, Precision>=0.90, Recall>=0.95',
        failures=fail,independent_external_test_images=0)
    c.js(root/'stage_gate.json',gate)
    # Metrics and training curves use fixed axes; background-dominated accuracy is not substituted for Dice.
    fig,axes=plt.subplots(1,4,figsize=(12,3.1));x=np.arange(3)
    for ax,key in zip(axes,['Precision','Recall','Dice','IoU']):
        for offset,(split,color) in enumerate([('train','#afbab5'),('val','#477d69'),('test','#7b6e91')]):
            ax.bar(x+(offset-1)*.24,[allmetrics[n][split]['root'][key] for n in NAMES],width=.24,color=color,label=split)
        ax.set_xticks(x,list(NAMES.values()),rotation=20,fontsize=8);ax.set_ylim(0,1);ax.set_title(key)
        if key in ['Recall','Dice']:ax.axhline(.95,color='#a34f4f',ls='--',lw=1)
    axes[0].legend(frameon=False,fontsize=7);savefig(fig,figures/'root_model_comparison')
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),sharey=True)
    for ax,name in zip(axes,NAMES):
        with (root/name/'history.csv').open(encoding='utf-8-sig') as f:history=list(csv.DictReader(f))
        hh=[r for r in history if r['validation_Dice']]
        for key,color in [('Dice','#477d69'),('Recall','#7b6e91'),('Precision','#be914d')]:
            ax.plot([int(r['epoch']) for r in hh],[float(r['validation_'+key]) for r in hh],label=key,c=color)
        ax.axvline(read(root/name/'training_run.json')['best_epoch'],c='#aaa',ls=':');ax.set(title=NAMES[name],xlabel='Epoch',ylim=(0,1))
    fig.suptitle('Validation at threshold0.5 — selected fine-tuning stage',fontsize=10)
    axes[0].legend(frameon=False,fontsize=7);savefig(fig,figures/'validation_learning_curves')
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    for j,name in enumerate(NAMES):
        pairs=[r for r in lengthrows if r['model']==name and r['split']=='test'];xx=np.array([r['human_label_length_px'] for r in pairs]);yy=np.array([r['predicted_length_px'] for r in pairs]);m=lengthmetrics[name]['test']
        ax=axes[0,j];ax.scatter(xx,yy,s=15,color='#477d69');bound=max(xx.max(),yy.max())*1.05
        ax.plot([0,bound],[0,bound],'--',c='#999',lw=1);ax.set(title=NAMES[name]+'\nR²={:.3f}; RMSE={:.1f}px'.format(m['R2_agreement'],m['RMSE_px']),xlabel='Human-label length (px)',ylabel='Predicted length (px)')
        ax=axes[1,j];ax.scatter((xx+yy)/2,yy-xx,s=15,color='#7b6e91')
        for key,style in [('mean_bias_px','-'),('Bland_Altman_lower_px','--'),('Bland_Altman_upper_px','--')]:ax.axhline(m[key],c='#999',ls=style,lw=1)
        ax.set(xlabel='Mean length (px)',ylabel='Prediction − reference (px)')
    savefig(fig,figures/'root_length_agreement')
    fig,ax=plt.subplots(figsize=(4.5,3.6))
    for name in NAMES:
        probs=np.concatenate([np.load(root/name/'predictions/test/C4'/(t+'_probability.npy')).ravel() for t in c.PERIODS]);gt=np.concatenate([data.gt[t][:,2070:2518].ravel() for t in c.PERIODS])
        points=[c.metrics(c.counts(gt,probs>=th)) for th in np.linspace(.01,.99,99)]
        ax.plot([q['Recall'] for q in points],[q['Precision'] for q in points],label=NAMES[name])
    ax.set(xlabel='Root recall',ylabel='Root precision',xlim=(0,1),ylim=(0,1));ax.legend(frameon=False,fontsize=8);savefig(fig,figures/'heldout_precision_recall')
    # Fixed worst-error windows expose remaining failures rather than selecting flattering examples.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for name in NAMES:
        canvas=Image.new('RGB',(1024,4*294),'white');draw=ImageDraw.Draw(canvas);threshold=read(root/name/'training_run.json')['frozen_threshold']
        for i,t in enumerate(c.PERIODS):
            gt=data.gt[t][:,2070:2518]==255;pr=np.load(root/name/'predictions/test/C4'/(t+'_probability.npy'))>=threshold
            x,y=max(c.COORDS,key=lambda xy:int((gt!=pr)[xy[1]:xy[1]+256,xy[0]:xy[0]+256].sum()))
            g=gt[y:y+256,x:x+256];p=pr[y:y+256,x:x+256];rgb=data.rgb[t][y:y+256,2070+x:2070+x+256]
            err=np.full_like(rgb,255);err[g&p]=[45,120,75];err[g&~p]=[37,99,235];err[~g&p]=[220,38,38]
            panels=[rgb,np.repeat((g*255).astype(np.uint8)[:,:,None],3,2),np.repeat((p*255).astype(np.uint8)[:,:,None],3,2),err]
            for j,(im,label) in enumerate(zip(panels,['RGB','Human mask','Predicted mask','FP red / FN blue'])):
                canvas.paste(Image.fromarray(im),(j*256,i*294+38));draw.text((j*256+3,i*294+6),t+' '+label,font=font,fill='black')
        canvas.save(figures/(name+'_heldout_errors.png'),dpi=(300,300))
    # Native full upper products from selected model are development visualizations, not test scores.
    products=root/'selected_root_products';products.mkdir(exist_ok=True)
    for t in c.PERIODS:
        rgb=data.rgb[t];mask=np.array(Image.open(root/winner/'predictions/upper'/(t+'_root_mask.png')))==255
        assert mask.shape==rgb.shape[:2]
        Image.fromarray(rgb).save(products/(t+'_original_RGB.png'))
        Image.fromarray(data.gt[t]).save(products/(t+'_human_root_mask.png'))
        Image.fromarray(mask.astype(np.uint8)*255).save(products/(t+'_predicted_root_mask.png'))
        rgba=np.concatenate([rgb,(mask.astype(np.uint8)*255)[:,:,None]],axis=2)
        rgba[~mask,:3]=0
        assert np.array_equal(rgba[mask,:3],rgb[mask]);Image.fromarray(rgba).save(products/(t+'_root_only_RGBA.png'))
        overlay=rgb.copy();overlay[mask]=(rgb[mask]*.5+np.array([20,160,80])*.5).astype(np.uint8)
        Image.fromarray(overlay).save(products/(t+'_root_overlay.png'),dpi=(300,300))
        probability=np.load(root/winner/'predictions/upper'/(t+'_probability.npy'))
        Image.fromarray(np.round(probability*65535).astype(np.uint16)).save(products/(t+'_root_probability_u16.png'))
    # A compact four-period comparison; separate native files retain every pixel.
    pw=900;ph=round(1108*pw/2541);rh=ph+48
    overview=Image.new('RGB',(pw*4,rh*4),'white');draw=ImageDraw.Draw(overview)
    label_font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    for i,t in enumerate(c.PERIODS):
        images=[Image.open(products/(t+'_original_RGB.png')).convert('RGB'),
            Image.fromarray(255-np.array(Image.open(products/(t+'_human_root_mask.png')))).convert('RGB'),
            Image.fromarray(255-np.array(Image.open(products/(t+'_predicted_root_mask.png')))).convert('RGB'),
            Image.open(products/(t+'_root_overlay.png')).convert('RGB')]
        for j,(im,label) in enumerate(zip(images,['Original RGB','Human root mask','Predicted root mask','Root prediction overlay'])):
            overview.paste(im.resize((pw,ph),Image.Resampling.LANCZOS),(j*pw,i*rh+48))
            draw.text((j*pw+12,i*rh+10),t+'  '+label,font=label_font,fill='#20352d')
    overview.save(figures/'four_period_root_background_overview.png',dpi=(300,300))
    val=allmetrics[winner]['val']['root'];test=allmetrics[winner]['test']['root'];lm=lengthmetrics[winner]['test']
    rows=['|模型|验证 Precision|验证 Recall|验证 Dice|验证 IoU|留出 Precision|留出 Recall|留出 Dice|留出 IoU|',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:rows.append('|'+NAMES[name]+'|'+'|'.join('{:.4f}'.format(allmetrics[name][s]['root'][k]) for s in ['val','test'] for k in ['Precision','Recall','Dice','IoU'])+'|')
    text='''# 0353-2-B3：第一阶段根／背景分割

仅训练根与背景二分类。新生、延伸、持续等属性完整保留在原始标注及切块中，本阶段无时序输入或时序损失；死亡不纳入。

输入为四期已配准裁剪拼接 RGB 的上部原像素区域，各2541×1108像素；下部及白色分隔带排除。本轮不重新配准或改变RGB。256×256原像素切块：训练120、验证40、空间留出测试40张；分别对应30、10、10个物理窗，每窗包含四期。跨组至少64像素隔离，组内重叠融合后每个像素只计一次。随机裁块及旋转增强不增加独立样本量。

'''+ '\n'.join(rows)+'''

选优仅依据验证集严格根像素 Dice：**'''+NAMES[winner]+'''**。留出区没有参与参数拟合或阈值选取，但该位置 C3/C4 已在早期预实验使用，因此不是全新独立测试。独立位置／独立原始图像测试数量均为0，不能据此声称跨位置泛化。

三模型计算预算不同，U-Net额外进行一次验证集选优的二分类精调，并调整困难负样本BCE权重；全部配置与比较记录已保存。学习曲线显示最终采用精调阶段的轮次。这是本预实验配置之间的比较，不能作为严格等预算、等损失的架构普遍优劣结论。

根分割门槛：验证和空间留出均 Dice≥0.95、Precision≥0.90、Recall≥0.95。当前'''+('达到根分割门槛。时序仍需独立实施和验证。' if not fail else '**尚未达到门槛，因此不启动时序训练，也不预测其他位置。**')+'''

人工笔迹原样保留，不膨胀、不自动补根。四期上部未涂画64按用户“已完成标注”的范围声明解释为背景；原始JSON保留全部像素状态。笔迹大多约1–3像素宽，严格 Dice 反映对现有笔迹的一致程度，并非完整根体边界准确率。未标注细根、笔迹偏移与真实识别错误都可能导致差异；误差图保留这些差异，不据此删改真值。

补充的标签几何敏感性图只描述人工笔迹和1像素自平移时的严格Dice，不是模型成绩或准确率上限，也不用于降低门槛。

背景合并为非根类，包含土壤、水珠、杂物。现有标注没有背景子类，因此不能分别给出水珠／土壤识别率。总体背景占比高，不能用总体像素准确率替代根类指标。clDice仅作为拓扑补充，不能替代严格 Dice 门槛。

选中模型在空间留出区的掩膜派生根长一致性：R²={:.4f}，RMSE={:.2f}像素。参考为人工掩膜骨架长度，不是独立人工量尺测量；未提供物理标尺，不能换算毫米。32对为8个空间块×4期，重复观察相关，不能称为32个独立生物学样本。R²采用1−SSE/SST，负值保留。

原始 RGB、人工根掩膜、预测根掩膜、保留根像素原色的透明根图、叠加图和16位根概率图均按原始像素保存。透明图将背景RGB置零、alpha置零，保留根像素原色；原始RGB另存。四期综合概览仅缩小显示，原像素单图不缩放。整幅图包含训练区域，仅供查看；正式指标来自隔离后的留出区域。图中的绿色只表示根预测，不表示新生或持续。

图表包括三模型对比、验证学习曲线、Precision–Recall曲线、根长一致性与Bland–Altman图、每期误差较大的原像素窗口。FP红色＝按现有标注的非根区预测为根，可能包含未标细根或边界差异；FN蓝色＝人工标记根漏检；TP绿色＝一致根像素。误差颜色表示与人工掩膜的差异，不是对水珠或土壤子类的人工鉴定。
'''.format(lm['R2_agreement'],lm['RMSE_px'])
    (root/'结果说明.md').write_text(text,encoding='utf-8');print('ROOT_REPORT_DONE',winner,'ROOT_GATE',not fail,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--mode',choices=['select','report'],required=True);a=p.parse_args()
    (select if a.mode=='select' else report)(a.run)
if __name__=='__main__':main()
