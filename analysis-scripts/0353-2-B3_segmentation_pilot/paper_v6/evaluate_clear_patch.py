#!/usr/bin/env python3
"""Frozen, complete-column evaluation of the clear-training-patch experiment."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import protocol as p
from train import RootUNet,inference
from figures import save,TEAL,GREY,DARK,panel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
from evaluate import length_field,agreement
from clear_patch_trial import read,csvout

NAMES={'v6_start':'Starting U-Net','control':'Continued / original sampling','screened':'Continued / clear-patch sampling'}

def training_figure(run,figdir):
 fig,axes=plt.subplots(1,2,figsize=(7.2,2.9),layout='constrained')
 for arm,color,style in [('control',GREY,'--'),('screened',TEAL,'-')]:
  with (run/arm/'history.csv').open(encoding='utf-8-sig') as f:history=[r for r in csv.DictReader(f) if r['validation_Dice']]
  epochs=[0]+[int(r['epoch']) for r in history];values=[read(run/arm/'initial_validation.json')['Dice']]+[float(r['validation_Dice']) for r in history]
  for ax in axes:ax.plot(epochs,values,c=color,ls=style,marker='o',ms=3,lw=1,label='Original sampler' if arm=='control' else 'Clear-patch sampler')
 for ax,title,limits,label in zip(axes,['Full score range','Expanded scale'],[(0,1),(.73,.76)],'ab'):
  ax.set(xlabel='Additional fine-tuning epoch',ylabel='Full C3 Dice at threshold0.5',ylim=limits,title=title,xticks=[0,5,10,15,20]);panel(ax,label)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
 save(fig,figdir/'Fig2_matched_training_curves')

def whole(data,infer):
 h,w=1108,2541;side=384;xs=list(range(0,w-side+1,192))+[w-side];ys=list(range(0,h-side+1,192))+[h-side]
 xs=sorted(set(xs));ys=sorted(set(ys));coords=[(x,y) for y in ys for x in xs]
 hann=np.hanning(side).astype(np.float32);blend=np.maximum(hann[:,None]*hann[None,:],.05);maps={}
 for t in p.PERIODS:
  sm=np.zeros((h,w),np.float32);wt=np.zeros_like(sm)
  for i in range(0,len(coords),2):
   xy=coords[i:i+2];ims=np.stack([data.rgb[t][y:y+side,x:x+side] for x,y in xy]).astype(np.float32)/255
   for (x,y),prob in zip(xy,infer(ims)):sm[y:y+side,x:x+side]+=prob*blend;wt[y:y+side,x:x+side]+=blend
  assert (wt>0).all();maps[t]=sm/wt
 return maps

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--previous',type=Path,required=True);a.add_argument('--threads',type=int,default=3);args=a.parse_args()
 run=args.run;data=p.Data(args.data);torch.set_num_threads(args.threads);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 start=read(args.previous/'unet_pretrained/validation_inference_selection.json')['selected']
 choices={'v6_start':dict(tta=start['tta'],threshold=start['threshold'],metrics=start['root'],checkpoint=str(args.previous/'unet_pretrained/best.pt'))}
 for arm in ['control','screened']:
  sel=read(run/arm/'validation_selection.json')['selected'];choices[arm]=dict(**sel,checkpoint=str(run/arm/'best.pt'))
 winner=max(choices,key=lambda n:choices[n]['metrics']['Dice'])
 selection=dict(winner=winner,candidates=choices,criterion='C3 full-column strict foreground Dice; existing starting model wins an exact tie',C4_used_for_selection=False,
  screening_improves_validation_over_control=choices['screened']['metrics']['Dice']>choices['control']['metrics']['Dice'],
  new_models_replace_published_v6_ensemble=False,note='This ablation evaluates one U-Net architecture; previous v6 ensemble remains a separate published configuration')
 p.js(run/'frozen_selection.json',selection)
 metrics={};rows=[];lengthrows=[];lengthstats={};saved_checks=0;infer_by_name={}
 for name,choice in choices.items():
  ck=torch.load(choice['checkpoint'],map_location=device,weights_only=True);model=RootUNet().to(device);model.load_state_dict(ck['state_dict']);infer=inference(model,device,choice['tta']);infer_by_name[name]=infer
  metrics[name]={};lengthstats[name]={}
  for split,col in [('val',3),('test',4)]:
   folder=run/'evaluation'/name/split;folder.mkdir(parents=True,exist_ok=True)
   if name=='v6_start':
    source=args.previous/'candidates/refined_unet_tta'/split
    if all((source/(t+'_probability.npy')).exists() for t in p.PERIODS):maps={t:np.load(source/(t+'_probability.npy')) for t in p.PERIODS}
    elif split=='val':maps={t:np.load(args.previous/'unet_pretrained/val_tta'/(t+'_probability.npy')) for t in p.PERIODS}
    else:maps=data.column(infer,col)
   elif split=='val':
    source=run/name/('val_tta' if choice['tta'] else 'val_identity');maps={t:np.load(source/(t+'_probability.npy')) for t in p.PERIODS}
   else:maps=data.column(infer,col)
   p.save_maps(folder,maps);score=data.evaluate(maps,col,choice['threshold']);metrics[name][split]=score
   if split=='val':assert score['root']==choice['metrics']
   rows.append(dict(configuration=name,partition=split,period='all',**score['root']))
   local=[];recount=np.zeros(4,np.int64)
   for t in p.PERIODS:
    pred=maps[t]>=choice['threshold'];gt=data.gt[t][:,p.COLS[col]:p.COLS[col]+448];g=gt==255
    for stem,mask in [('prediction',pred),('human',g)]:Image.fromarray(mask.astype(np.uint8)*255).save(folder/(t+'_'+stem+'.png'))
    reload=np.array(Image.open(folder/(t+'_prediction.png')))==255;recount+=p.counts(gt,reload);saved_checks+=1
    rows.append(dict(configuration=name,partition=split,period=t,**score['per_period'][t]));lf=length_field(g);lp=length_field(pred)
    for yi in range(4):
     for xi in range(2):
      sl=(slice(yi*277,(yi+1)*277),slice(xi*224,(xi+1)*224));row=dict(configuration=name,partition=split,period=t,block=f'X{xi}Y{yi}',human_length_px=float(lf[sl].sum()),predicted_length_px=float(lp[sl].sum()))
      local.append(row);lengthrows.append(row)
   assert p.metrics(recount)==score['root'];lengthstats[name][split]=agreement([r['human_length_px'] for r in local],[r['predicted_length_px'] for r in local])
  print('CLEAR_PATCH_EVALUATED',name,metrics[name]['val']['root'],metrics[name]['test']['root'],flush=True)
 p.js(run/'all_metrics.json',metrics);p.js(run/'root_length_agreement.json',lengthstats);csvout(run/'metrics.csv',rows);csvout(run/'root_length_pairs.csv',lengthrows)
 products=run/'images';products.mkdir(exist_ok=True)
 if winner=='v6_start' and (args.previous/'whole_cache/refined_unet_tta').exists():maps={t:np.load(args.previous/'whole_cache/refined_unet_tta'/(t+'_probability.npy')) for t in p.PERIODS}
 else:maps=whole(data,infer_by_name[winner])
 for t in p.PERIODS:
  rgb=data.rgb[t];mask=maps[t]>=choices[winner]['threshold'];rgba=np.concatenate([rgb,np.uint8(mask[:,:,None])*255],axis=2);rgba[~mask,:3]=0
  overlay=rgb.copy();overlay[mask]=(rgb[mask]*.5+np.array([0,133,125])*.5).astype(np.uint8)
  for stem,im in [('RGB',rgb),('human',data.gt[t]),('prediction',mask.astype(np.uint8)*255),('root_only',rgba),('overlay',overlay)]:Image.fromarray(im).save(products/(t+'_'+stem+'.png'))
  assert np.array_equal(np.array(Image.open(products/(t+'_RGB.png'))),data.rgb[t]);assert np.array_equal(np.array(Image.open(products/(t+'_human.png'))),data.gt[t]);saved_checks+=2
 figdir=run/'figures';figdir.mkdir(exist_ok=True)
 fig,axes=plt.subplots(1,4,figsize=(7.2,2.8),layout='constrained',sharey=True)
 for ax,key,label in zip(axes,['Precision','Recall','Dice','IoU'],'abcd'):
  for j,n in enumerate(NAMES):
   v=metrics[n]['val']['root'][key];test=metrics[n]['test']['root'][key];color=TEAL if n=='screened' else GREY
   ax.plot([v,test],[j,j],c=color,lw=.8);ax.scatter(v,j,facecolors='white',edgecolors=color,s=25);ax.scatter(test,j,c=color,marker='D',s=25)
  ax.set(xlim=(0,1),xlabel=key,xticks=[0,.25,.5,.75,1]);ax.grid(axis='x',c='#E5E7EB',lw=.5);panel(ax,label)
 axes[0].set_yticks(range(3),['Starting U-Net','Original sampler','Clear-patch sampler']);axes[0].invert_yaxis()
 fig.legend(handles=[Line2D([],[],marker='o',ls='none',mfc='white',mec=DARK,label='Validation C3'),Line2D([],[],marker='D',ls='none',mfc=DARK,mec=DARK,label='Repeated C4')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,figdir/'Fig1_clear_patch_ablation')
 training_figure(run,figdir)
 fig,axes=plt.subplots(4,3,figsize=(7.2,4.65),layout='constrained')
 for i,t in enumerate(p.PERIODS):
  rgb=data.rgb[t];gt=data.gt[t]==255;mask=np.array(Image.open(products/(t+'_prediction.png')))==255
  for j,image in enumerate([rgb,np.where(gt[:,:,None],np.array([55,65,81],np.uint8),255),np.where(mask[:,:,None],np.array([0,133,125],np.uint8),255)]):axes[i,j].imshow(image.astype(np.uint8),interpolation='none');axes[i,j].set_axis_off()
  axes[i,0].text(-.025,.5,t,ha='right',va='center',transform=axes[i,0].transAxes,fontweight='bold')
 for ax,title in zip(axes[0],['Original RGB','Human reference','Selected U-Net']):ax.set_title(title)
 save(fig,figdir/'Fig3_four_period_qualitative',True)
 # The source provenance still verifies every original RGB/mask/attribute file.
 p.Data(args.data)
 with (run/'final_training_selection.csv').open(encoding='utf-8-sig') as f:manifest=list(csv.DictReader(f))
 kept=[r for r in manifest if r['final_selected']=='True'];assert len(kept)==70 and all(int(r['column'])<3 for r in kept)
 p.js(run/'verification.json',dict(passed=True,saved_prediction_and_RGB_checks=saved_checks,all_original_provenance_hashes_unchanged=True,
  training_selected_root_patches=len(kept),validation_or_C4_filtered=False,selection_uses_C4=False,original_progress_SHA256=p.sha(args.data/'original_human_progress.json'),
  training_manifest_SHA256=p.sha(run/'final_training_selection.csv'),frozen_selection_SHA256=p.sha(run/'frozen_selection.json')))
 p.js(run/'stage_gate.json',dict(near_one_gate_passed=all(metrics[winner][sp]['root']['Dice']>=.95 and metrics[winner][sp]['root']['Recall']>=.95 and metrics[winner][sp]['root']['Precision']>=.9 for sp in ['val','test']),
  temporal_stage_started=False,death_included=False,other_positions_predicted=False,external_independent_test_images=0))
 table='|方案|验证 Dice|留出 Precision|留出 Recall|留出 Dice|留出 IoU|\n|---|---:|---:|---:|---:|---:|\n'
 for n in NAMES:
  v=metrics[n]['val']['root'];m=metrics[n]['test']['root'];table+='|'+{'v6_start':'起始预训练 U-Net','control':'继续训练／原采样','screened':'继续训练／清晰图块'}[n]+'|'+f"{v['Dice']:.4f}|{m['Precision']:.4f}|{m['Recall']:.4f}|{m['Dice']:.4f}|{m['IoU']:.4f}|\n"
 verdict='筛选组的验证 Dice 高于原采样对照组。' if selection['screening_improves_validation_over_control'] else '筛选组的验证 Dice 未超过原采样对照组，不能声称筛选带来提分。'
 report=f'''# 0353-2-B3 清晰训练图块对照 v7

只筛选训练图块，原始 RGB 与人工掩膜不变。候选根系窗口120张，经训练区局部可见度初筛保留84张，再经 RGB 复核剔除14张，最终70张根系窗口；8张无人工根像素背景候选复核后保留4张。局部可见度只是筛选依据，并非自动判定的人工标注置信度，未保留的图块仍可包含真根。四期同一空间组的划分保持不变。

{table}
{verdict}按完整 C3 的严格 Dice 选中的单模型配置为 **{winner}**。完整验证和 C4 低对比区域均保留，没有通过只评价容易区域提高分数。本轮是 U-Net 采样消融，不是重新训练三种架构，也未替换已发布的 v6 集成配置。全部结果保留，不隐去退步。

两组从同一 v6 第35轮权重开始，以相同随机种子、学习率、损失函数、20轮、每轮32样本／批次2继续训练，共320次优化器更新。对照组保留原384原像素动态采样，筛选组使用固定清晰窗口及约10%的128原像素背景批次，均不缩放；因此更新次数相同，但像素数与计算量不完全相同。第0轮也参与验证择优，防止继续训练反而降低效果。阈值和翻转平均只用 C3 选择；回归区块保留四期。仅一个种子、一个位置，不能把差异解释为已证实的普遍规律。

训练源仍仅1个位置／4张上部2541×1108 RGB。图块重叠，不是70个独立生物样本。验证与空间留出仍各40张导出窗口，正式指标在完整448像素列融合后计数一次。C4已在多轮中查看，是同位置重复探索性基准，外部独立测试图像0。背景为合并类别，没有独立土壤／水珠／杂物真值。没有编辑真值、膨胀掩膜、删除难评价像素或以目标分数反向修订。

根长结果是人工笔刷掩膜与预测掩膜的骨架长度一致性，单位原像素，非独立人工测量；32区块—时期对存在相关性。原始分割指标为主要评价，不把较高根长相关性替代像素 Dice。没有预设达到0.9或进入时序模型。全景图包含训练像素，仅作定性观察，正式数值来自隔离列。

图1采样对照指标；图2同预算学习曲线；图3验证选中单模型四期全景。提供600dpi PNG、TIFF、PDF，定量图另有SVG。70张训练窗口的坐标、入选／剔除依据、原始人工进度、完整脚本、权重、逐时期指标和可复算掩膜均已保存。
'''
 (run/'Results_CN.md').write_text(report,encoding='utf-8');print('CLEAR_PATCH_COMPLETE',winner,verdict,flush=True)
if __name__=='__main__':main()
