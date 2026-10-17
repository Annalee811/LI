#!/usr/bin/env python3
"""Native continuity products, bounded centerline display, figures and ZIPs."""
import argparse,json,shutil,zipfile,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import protocol as p
from continuity_v8 import proposals,apply_links,read,csvout
from evaluate_continuity_v8 import paths
from figures import save,mask_rgb,TEAL,DARK,GREY,RED,panel

def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
def svg(mask,path):
 h,w=mask.shape;lines=[];maximum=0.;count=0
 for track in paths(mask):
  xy=track[:,::-1].astype(float);smooth=ndi.gaussian_filter1d(xy,1.,axis=0,mode='nearest');delta=smooth-xy;norm=np.linalg.norm(delta,axis=1)
  delta*=np.minimum(1.,.5/np.maximum(norm,1e-12))[:,None];smooth=xy+delta;smooth[0]=xy[0];smooth[-1]=xy[-1]
  maximum=max(maximum,float(np.linalg.norm(smooth-xy,axis=1).max()));coords=' '.join(f'{x:.3f},{y:.3f}' for x,y in smooth)
  lines.append('<polyline points="'+coords+'"/>');count+=1
 text=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>Root centerline display; at most0.5 native pixel smoothing; not a segmentation mask</title><g fill="none" stroke="{TEAL}" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">'+''.join(lines)+'</g></svg>\n'
 path.write_text(text);return dict(paths=count,maximum_display_displacement_native_px=maximum,quantitative_mask_altered=False)

def zipout(path,entries):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for source,name in entries:
   if source.is_dir():
    for f in sorted(source.rglob('*')):
     if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(name/f.relative_to(source)))
   else:z.write(source,str(name))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 assert path.stat().st_size<100*1024**2
 return dict(file=path.name,bytes=path.stat().st_size,SHA256=p.sha(path),CRC_verified=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--whole',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 data=p.Data(args.data);run=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True);selection=read(run/'frozen_selection.json');cfg=selection['configuration'];classifier=read(run/'link_classifier.json');metrics=read(run/'all_metrics.json')
 images=run/'images';images.mkdir(exist_ok=True);native=run/'native_dataset';native.mkdir(exist_ok=True);smooth_stats={};full_links={};whole_counts={}
 copy(args.data/'original_human_progress.json',native/'original_human_progress.json')
 for t in p.PERIODS:
  rgb=data.rgb[t];prob=np.load(args.whole/(t+'_probability.npy'));base,candidates=proposals(prob,rgb);after,added,links=apply_links(base,candidates,cfg,classifier) if cfg else (base,np.zeros_like(base),[])
  assert np.all(after[base])
  for layer,im in [('RGB',rgb),('mask',data.gt[t])]:
   target=native/'dataset/upper'/(t+'_'+layer+'.png');target.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(im).save(target);assert np.array_equal(np.array(Image.open(target)),im)
  for stem,im in [('before',base),('after',after),('added',added)]:Image.fromarray(im.astype(np.uint8)*255).save(images/(t+'_'+stem+'_mask.png'))
  rgba=np.concatenate([rgb,after[:,:,None].astype(np.uint8)*255],axis=2);rgba[~after,:3]=0;assert np.array_equal(rgba[after,:3],rgb[after])
  Image.fromarray(rgba).save(images/(t+'_root_only_RGBA.png'))
  overlay=rgb.copy();overlay[after]=(rgb[after]*.5+np.array([0,133,125])*.5).astype(np.uint8);overlay[added]=(rgb[added]*.5+np.array([213,94,0])*.5).astype(np.uint8)
  Image.fromarray(overlay).save(images/(t+'_overlay.png'))
  smooth_stats[t]=dict(prediction=svg(after,images/(t+'_prediction_smooth_centerline.svg')),human_reference=svg(data.gt[t]==255,images/(t+'_human_smooth_centerline.svg')))
  p.js(images/(t+'_added_links.json'),links);full_links[t]=links;whole_counts[t]=dict(links=len(links),added_pixels=int(added.sum()),before_pixels=int(base.sum()),after_pixels=int(after.sum()))
  print('NATIVE_CONTINUITY',t,whole_counts[t],flush=True)
 files={str(f.relative_to(native)):p.sha(f) for f in sorted((native/'dataset/upper').glob('*.png'))};p.js(native/'dataset/provenance.json',dict(files=files,source_progress_sha256=p.sha(native/'original_human_progress.json'),note='Minimal lossless RGB/mask snapshot; temporal attributes remain in the original progress JSON'))
 p.js(run/'smooth_display_verification.json',smooth_stats);p.js(run/'whole_field_changes.json',whole_counts)
 # Show every accepted formal-column link, including those with poor reference overlap.
 details=run/'all_link_details';details.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
 for split,col in [('val',3),('test',4)]:
  folder=run/'evaluation'/split
  for t in p.PERIODS:
   gt=np.array(Image.open(folder/(t+'_human.png')))==255;before=np.array(Image.open(folder/(t+'_before.png')))==255;after=np.array(Image.open(folder/(t+'_after.png')))==255;added=after&~before
   links=read(folder/(t+'_accepted_links.json'))
   for i,link in enumerate(links):
    center=np.mean(link['endpoints'],axis=0).astype(int);y=int(np.clip(center[0]-48,0,1012));x=int(np.clip(center[1]-48,0,352));sl=(slice(y,y+96),slice(x,x+96));rgb=data.rgb[t][y:y+96,p.COLS[col]+x:p.COLS[col]+x+96]
    parts=[rgb,mask_rgb(gt[sl],DARK),mask_rgb(before[sl],DARK),mask_rgb(after[sl]),mask_rgb(added[sl],RED)]
    canvas=Image.new('RGB',(5*144,174),'white');draw=ImageDraw.Draw(canvas)
    for j,(im,title) in enumerate(zip(parts,['RGB','Human','Before','After','Added'])):
     canvas.paste(Image.fromarray(im).resize((144,144),Image.Resampling.NEAREST),(j*144,30));draw.text((j*144+3,7),title,font=font,fill='black')
    canvas.save(details/(split+'_'+t+f'_link_{i:03d}.png'))
 # One common C3 window selected by added-pixel count, not reference correctness.
 total=sum(np.array(Image.open(run/'evaluation/val'/(t+'_added.png')))>0 for t in p.PERIODS);best=(-1,0,0)
 for y in range(0,981,32):
  for x in range(0,321,32):
   n=int(total[y:y+128,x:x+128].sum())
   if n>best[0]:best=(n,y,x)
 _,y,x=best;sl=(slice(y,y+128),slice(x,x+128));figdir=run/'figures';figdir.mkdir(exist_ok=True)
 fig,axes=plt.subplots(4,5,figsize=(7.2,5.8),layout='constrained')
 for i,t in enumerate(p.PERIODS):
  folder=run/'evaluation/val';rgb=data.rgb[t][y:y+128,1558+x:1558+x+128]
  arrays=[rgb]+[mask_rgb(np.array(Image.open(folder/(t+'_'+s+'.png')))[sl]>0,c) for s,c in [('human',DARK),('before',DARK),('after',TEAL),('added',RED)]]
  for j,im in enumerate(arrays):axes[i,j].imshow(im,interpolation='none');axes[i,j].set_axis_off();panel(axes[i,j],chr(97+i*5+j))
  axes[i,0].text(-.04,.5,t,transform=axes[i,0].transAxes,ha='right',va='center',fontweight='bold')
 for ax,title in zip(axes[0],['RGB','Human reference','Before','After','Added links']):ax.set_title(title,pad=12)
 save(fig,figdir/'Fig1_fixed_validation_detail',True)
 fig,axes=plt.subplots(4,3,figsize=(7.2,4.65),layout='constrained')
 for i,t in enumerate(p.PERIODS):
  arrays=[mask_rgb(data.gt[t]==255,DARK)]+[mask_rgb(np.array(Image.open(images/(t+'_'+s+'_mask.png')))>0,c) for s,c in [('before',DARK),('after',TEAL)]]
  for j,im in enumerate(arrays):axes[i,j].imshow(im,interpolation='none');axes[i,j].set_axis_off()
  axes[i,0].text(-.04,.5,t,transform=axes[i,0].transAxes,ha='right',va='center',fontweight='bold')
 for ax,title in zip(axes[0],['Human reference','Before','After']):ax.set_title(title)
 save(fig,figdir/'Fig2_native_four_period_masks',True)
 fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained')
 for ax,key,label in zip(axes,['reference_pair_connectivity','skeleton_endpoints','Dice'],'abc'):
  for j,(mode,color) in enumerate([('before',GREY),('after',TEAL)]):
   values=[metrics[sp][mode]['root' if key=='Dice' else 'topology'][key] for sp in ['val','test']]
   bars=ax.bar(np.arange(2)+(j-.5)*.3,values,width=.28,color=color,label=mode.capitalize())
   for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+(8 if key=='skeleton_endpoints' else .015),str(v) if key=='skeleton_endpoints' else f'{v:.3f}',ha='center',fontsize=6)
  ax.set(xticks=[0,1],xticklabels=['C3 validation','Repeated C4'],ylabel={'reference_pair_connectivity':'Reference-pair connectivity','skeleton_endpoints':'Skeleton endpoints','Dice':'Strict foreground Dice'}[key],ylim=(0,650) if key=='skeleton_endpoints' else (0,1.08));panel(ax,label)
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False);save(fig,figdir/'Fig3_continuity_and_accuracy')
 # Reproduce all strict foreground metrics from actual saved PNG masks.
 checks=0
 for split,col in [('val',3),('test',4)]:
  for mode in ['before','after']:
   cc=np.zeros(4,np.int64)
   for t in p.PERIODS:
    mask=np.array(Image.open(run/'evaluation'/split/(t+'_'+mode+'.png')))>0;cc+=p.counts(data.gt[t][:,p.COLS[col]:p.COLS[col]+448],mask);checks+=1
   assert p.metrics(cc)==metrics[split][mode]['root']
 p.Data(args.data);p.js(run/'export_verification.json',dict(passed=True,saved_mask_count_checks=checks,original_RGB_and_manual_masks_unchanged=True,all_baseline_root_pixels_retained=True,
  SVG_centerline_display_only=True,maximum_SVG_vertex_displacement_native_px=.5,selection_uses_C4=False))
 val=metrics['val'];test=metrics['test'];c=classifier;counts=read(run/'source.json');tables='|指标|连接前|连接后|\n|---|---:|---:|\n'
 for key,title in [('Precision','Precision'),('Recall','Recall'),('Dice','Dice'),('IoU','IoU')]:tables+=f"|{title}|{test['before']['root'][key]:.4f}|{test['after']['root'][key]:.4f}|\n"
 for key,title in [('reference_pair_connectivity','人工根段参考点对连通覆盖率'),('clDice_mean','平均 clDice')]:tables+=f"|{title}|{test['before']['topology'][key]:.4f}|{test['after']['topology'][key]:.4f}|\n"
 m=test['after']['topology']
 report=f'''# 0353-2-B3 根系断点与分叉连接优化 v8

使用当前 v7 起始预训练 U-Net 的原始四翻转得分（阈值0.55），保留原有根像素，新增短距离连接。新增轻量逻辑回归连接判别器，从训练区人工标注学习端点走向、分叉、距离、得分及RGB局部证据；根分割网络权重不变。训练样例为{c['positive']}个支持连接、{c['negative']}个不支持连接的衍生样例，来自四期×三个训练列，不是新增独立图像。不确定训练连接未作为确定标签。

同时考虑端点对端点和支根对主根中段，原图模型得分支持路径才可补接；避免绕行、循环以及顺带接上第三条根。连接判别器在C0–C2训练，仅用C3选择参数，保留原人工真值；没有直接复制人工线条到预测。选择依据为人工非分叉根段上参考点对的连通覆盖，约束严格 Dice、Precision、clDice 的变化。完整网格、判定器系数和逐条连接记录均保存。

## 重复空间留出 C4 的实测结果

{tables}
四期合计新增{m['accepted_links']}条连接、{m['added_pixels']}个像素；其中{m['added_reference_root_pixels']}个落在人工根像素、{m['added_reference_background_pixels']}个落在参考背景像素。不能把每条连接都认定为已确认正确，所有新增像素独立保存，并提供每条正式评价连接的原图／人工／前后／新增对照。

“参考点对连通覆盖”是在人工同一非分叉骨架段上，每4步取一对相隔8步的参考点，检查其1像素邻域的预测连通域是否相交；它是专门的拓扑诊断，不能代替严格像素Dice，也不代表已识别独立生物根实例。C4是同位置已多次查看的重复探索性空间留出，独立外部测试图像为0。整个v8迭代包含先试端点连接、再补充分叉连接和第三根限制；参数只按C3选择，但不能宣称新独立测试。根长结果仍为人工笔刷掩膜骨架与预测骨架的一致性，原像素单位、无独立人工长度测量，区块—时期对相关。

## 文件与显示方式

四期保留原像素RGB与人工掩膜，提供连接前／后掩膜、新增连接掩膜、原色透明根系、叠加图、全部坐标与分数。青绿色为根，叠加图橙色仅表示算法补接，不表示新生根。全上部图包含训练像素，仅供定性观察；正式指标来自隔离的C3/C4列。

另附预测及人工参考的平滑中心线SVG：只做不超过0.5原像素的显示平滑，端点与分叉位置固定；统一线宽不表示真实根直径，不参与Dice或根长测量。用于定量分析的二值掩膜没有平滑、腐蚀或加粗。原始RGB没有改色或提高信息分辨率。600dpi PNG／LZW TIFF／PDF及定量SVG供论文排版，不等同于模型已达高精度。

Fig1同一128像素验证窗口四期细节，窗口按新增像素数选取而非人工重合度，原图起点({1558+x},{y})；Fig2四期全景掩膜；Fig3连续性与分割指标。原始人工进度JSON完整保留。死亡与时序分类不纳入，未预测其他位置。
'''
 (run/'Results_CN.md').write_text(report,encoding='utf-8')
 (run/'Figure_captions_EN.md').write_text('Fig1. Fixed native128×128 C3 detail at all four dates, selected by the number of added pixels, not reference overlap. Orange denotes algorithmic gap-filling, not biological new roots. Fig2. Whole-upper-field reference/before/after masks, including training pixels; qualitative only. Fig3. Reference-segment anchor connectivity (1px neighborhoods), endpoint counts and strict pixel Dice on complete isolated C3 and repeatedly inspected C4 columns. Topology scores are not replacements for pixel accuracy or independent validation. Smoothed SVGs are separate centerline displays with≤0.5px vertex displacement and fixed junctions; quantitative masks are unchanged by display smoothing.\n')
 # Reusable three-archive layout: data root is directly accepted by protocol.Data.
 basepath=Path('0353-2-B3_continuity_v8');src=Path(__file__).resolve().parent
 report_files=['Results_CN.md','Figure_captions_EN.md','source.json','link_classifier.json','training_link_audit.csv','validation_grid.csv','frozen_selection.json','all_metrics.json','metrics.csv','root_length_agreement.json','root_length_pairs.csv','verification.json','export_verification.json','smooth_display_verification.json','whole_field_changes.json']
 if (run/'link_audit.csv').exists():report_files+=['link_audit.csv']
 entries=[(run/f,basepath/'evaluation'/f) for f in report_files]
 entries+=[(src/f,basepath/'source/paper_v6'/f) for f in ['continuity_v8.py','evaluate_continuity_v8.py','deliver_continuity_v8.py','protocol.py','train.py','figures.py','requirements.txt']]
 entries+=[(src.parent/'root_background_v5'/f,basepath/'source/root_background_v5'/f) for f in ['evaluate.py','common.py']]
 manifests=[]
 manifests.append(zipout(out/'0353-2-B3_continuity_v8_images.zip',[(native,basepath),(images,basepath/'images')]+[(run/'Results_CN.md',basepath/'README.md')]))
 figure_entries=entries+[(figdir,basepath/'figures'),(details,basepath/'all_link_details')]
 for split in ['val','test']:
  for f in (run/'evaluation'/split).glob('*'):
   if f.suffix!='.npy':figure_entries.append((f,basepath/'evaluation'/split/f.name))
 manifests.append(zipout(out/'0353-2-B3_continuity_v8_comparison.zip',figure_entries))
 score_entries=[(run/'train_probabilities',basepath/'train_probabilities'),(args.whole,basepath/'full_probabilities'),(run/'training_link_features.npz',basepath/'training_link_features.npz')]
 for split in ['val','test']:
  for f in (run/'evaluation'/split).glob('*.npy'):score_entries.append((f,basepath/'v7_snapshot/evaluation/v6_start'/split/f.name))
 manifests.append(zipout(out/'0353-2-B3_continuity_v8_probabilities.zip',score_entries))
 p.js(out/'SHA256.json',dict(archives=manifests,original_progress_SHA256=p.sha(args.data/'original_human_progress.json'),manual_annotations_unchanged=True))
 links='\n## 下载 ZIP\n\n- [四期原图、前后掩膜与平滑中心线](0353-2-B3_continuity_v8_images.zip)\n- [对照图片、全部连接细节、指标与完整脚本](0353-2-B3_continuity_v8_comparison.zip)\n- [完整原始概率与训练连接特征](0353-2-B3_continuity_v8_probabilities.zip)\n'
 reproduce='\n复现：三个ZIP解压合并到同一目录。根目录作为 DATA 和 RUN，先使用 `source/paper_v6/evaluate_continuity_v8.py --data "$DATA" --run "$RUN" --baseline "$RUN/v7_snapshot"` 复算，再使用 `source/paper_v6/deliver_continuity_v8.py --data "$DATA" --run "$RUN" --whole "$RUN/full_probabilities" --output /tmp/B3_continuity_repeat` 重建图包。将 evaluation/ 下的 link_classifier.json、source.json、frozen_selection.json等配置复制到RUN根目录。依赖沿用v6；原始NN权重仍为v6模型包中unet_pretrained/best.pt，连接判别器由本包JSON给出，不需要重新训练根网络。\n'
 (out/'README.md').write_text(report+links+reproduce,encoding='utf-8');copy(run/'metrics.csv',out/'metrics.csv');copy(run/'export_verification.json',out/'verification.json')
 for m in manifests:print('CONTINUITY_ARCHIVE_VERIFIED',m,flush=True)
if __name__=='__main__':main()
