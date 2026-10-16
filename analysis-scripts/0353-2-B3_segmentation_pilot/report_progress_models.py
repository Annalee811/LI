#!/usr/bin/env python3
"""Audit the new pilot, preserve human work, and make honest model comparison plots."""
from pathlib import Path
import argparse,copy,json,csv,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);a=p.parse_args();root=a.run
checks=[]
def check(name,condition):
 checks.append({'name':name,'passed':bool(condition)})
 if not condition:raise AssertionError(name)
def rle(array):
 data=array.ravel();starts=np.r_[0,np.flatnonzero(data[1:]!=data[:-1])+1];ends=np.r_[starts[1:],len(data)];out=np.column_stack((data[starts],ends-starts)).ravel().tolist();return out
selection=json.loads((root/'three_model_selection.json').read_text());temporal=json.loads((root/'temporal/metrics.json').read_text());scope=json.loads((root/'scope.json').read_text());predictions=json.loads((root/'four_period_prediction_summary.json').read_text());original=json.loads((root/'original_human_progress.json').read_text());project=copy.deepcopy(original)
check('winner selected only from validation scores',selection['winner']==max(selection['scores'],key=lambda x:x['validation_Dice'])['model_id'])
check('same training data for all three models',len({selection['manifest_sha256'],json.loads((root/'segformer/training/training_run.json').read_text())['manifest_sha256'],hashlib.sha256((a.dataset/'dataset/manifest.csv').read_bytes()).hexdigest()})==1)
with (a.dataset/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
check('T1 and T2 always same spatial group',all(len({q['split'] for q in rows if q['window_id']==r['window_id']})==1 for r in rows))
for phase in ['T1','T2','T3','T4']:
 folder=root/'four_period_predictions'/phase;rgb=np.array(Image.open(folder/'original_RGB_upper.png'));mask=np.array(Image.open(folder/'root_mask.png'));state=np.array(Image.open(folder/'temporal_class_with_undetermined.png'))
 check(phase+' native RGB/mask/state dimensions correct',rgb.shape==(1108,2541,3) and mask.shape==state.shape==(1108,2541))
 check(phase+' states only inside predicted roots',not ((state!=0)&(mask!=255)).any())
 check(phase+' correct class codes',set(np.unique(state)).issubset({0,1,2,3,4,255}))
 if phase=='T1':check('T1 all predicted roots are baseline, never newly born',np.array_equal(state==4,mask==255))
for item in project['images']:
 phase=item['name'][:2]
 if phase in ['T1','T2']:
  check(phase+' all original manual labels retained exactly',item==next(x for x in original['images'] if x['name']==item['name']));continue
 layers={k:np.full((item['height'],item['width']),64 if k=='mask' else 0,np.uint8) for k in ['mask','color','temporal','viability']};folder=root/'four_period_predictions'/phase
 mask=np.array(Image.open(folder/'root_mask.png'))==255;state=np.array(Image.open(folder/'temporal_class_with_undetermined.png'))
 layers['mask'][:1108][mask]=255;expected_temporal=np.zeros(mask.shape,np.uint8)
 for model_code,editor_code in [(1,3),(2,1),(3,2)]:expected_temporal[mask&(state==model_code)]=editor_code
 layers['temporal'][:1108]=expected_temporal
 check(phase+' editable roots equal model prediction with no old strokes',np.array_equal(layers['mask'][:1108]==255,mask))
 check(phase+' editable classes equal model prediction with no old attributes',np.array_equal(layers['temporal'][:1108],expected_temporal) and not layers['color'].any() and not layers['viability'].any())
 check(phase+' excluded lower region remains unlabelled',np.all(layers['mask'][1108:]==64) and not layers['temporal'][1108:].any())
 item['layers']={k:rle(v) for k,v in layers.items()};item['reviewed']=False;item['evidence']='';item['label_provenance']='Entirely model-predicted; previous T3/T4 strokes ignored per user; unpredicted pixels stay unknown64'
project['model_prediction_provenance']={'root_model':selection['winner'],'temporal_model':'paired RGB and frozen root probabilities','original_manual_progress_sha256':hashlib.sha256((root/'original_human_progress.json').read_bytes()).hexdigest(),'original_T1_T2_unchanged':True,'T3_T4_previous_strokes_ignored':True,'T3_T4_require_review':True,'source_images':'Select original four 2541x1867 full RGB files, then import this JSON into annotate.html'}
(root/'editable_T1T2_manual_T3T4_model.json').write_text(json.dumps(project,ensure_ascii=False,separators=(',',':')))
# Reconstruct the user's original palette directly from the uploaded RLE, never from predictions.
human_dir=root/'original_manual_label_review';human_dir.mkdir(exist_ok=True);human_records=[]
palette={0:[37,99,235],1:[22,163,74],2:[234,179,8],3:[147,51,234]}
for item in original['images']:
 phase=item['name'][:2]
 if phase not in ['T1','T2']:continue
 layers={k:np.repeat(np.asarray(v[::2],np.uint8),v[1::2]).reshape(item['height'],item['width'])[:1108] for k,v in item['layers'].items()}
 rgb=np.array(Image.open(root/'four_period_predictions'/phase/'original_RGB_upper.png'));overlay=rgb.copy();root_pixels=layers['mask']==255;colored=np.full_like(rgb,255)
 for code,color in palette.items():
  selected=root_pixels&(layers['temporal']==code);overlay[selected]=color;colored[selected]=color
 Image.fromarray(overlay).save(human_dir/f'{phase}_original_manual_RGB_overlay.png')
 Image.fromarray(colored).save(human_dir/f'{phase}_original_manual_labels_white.png')
 for key,array in layers.items():Image.fromarray(array).save(human_dir/f'{phase}_original_manual_{key}.png')
 check(phase+' original annotation review root pixels exactly match training mask',np.array_equal(layers['mask']==255,np.array(Image.open(a.dataset/f'dataset/upper/{phase}_mask.png'))==255))
 check(phase+' original annotation temporal classes exactly match training',np.array_equal(layers['temporal'],np.array(Image.open(a.dataset/f'dataset/upper/{phase}_temporal.png'))))
 human_records.append({'period':phase,'root_pixels':int(root_pixels.sum()),'temporal_root_pixels':{str(code):int((root_pixels&(layers['temporal']==code)).sum()) for code in palette}})
(human_dir/'original_label_counts.json').write_text(json.dumps({'source':'original_human_progress.json, direct lossless RLE decoding','roi_xyxy':[0,0,2541,1108],'editor_temporal_encoding':{'0':'unclassified; all T1 roots treated as baseline','1':'newly observed, green','2':'elongating, yellow','3':'persistent, purple'},'palette_rgb':palette,'periods':human_records},indent=2)+'\n')
check('no complete T3/T4 labels used for training',scope['T3_T4_used_as_negative_labels'] is False and json.loads((root/'temporal/config.json').read_text())['T3_T4_unreviewed_labels_used'] is False)
growth=next(r for r in temporal['test']['per_class'] if r['name']=='elongating');check('no fabricated elongating-root test score without labels',growth['support_pixels']==0 and growth['Dice'] is None and growth['Recall'] is None)
# Third model's actual stored held-out masks reproduce the reported confusion counts.
for split,key,col in [('val','validation',3),('test','spatial_holdout_test',4)]:
 total=np.zeros(4,np.int64)
 for phase in ['T1','T2']:
  gt=np.array(Image.open(root/f'segformer/predictions/{split}/{phase}_C{col}_human_mask.png'))==255;pr=np.array(Image.open(root/f'segformer/predictions/{split}/{phase}_C{col}_prediction.png'))==255
  total+=np.array([(gt&pr).sum(),(~gt&pr).sum(),(gt&~pr).sum(),(~gt&~pr).sum()],np.int64)
 metric=json.loads((root/'segformer/metrics.json').read_text())[key]['micro_metrics'];check('SegFormer '+split+' scores reproduce and pixels counted once',all(int(metric[k])==int(v) for k,v in zip(['TP','FP','FN','TN'],total)) and total.sum()==992768)
report={'passed':len(checks),'failed':0,'checks':checks,'scope':'Software, provenance and actual score checks, not independent anatomical annotation adjudication'};(root/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
colors=['#467565','#746b91','#4c7f9b'];labels=['U-Net','DeepLabV3+','SegFormer-B0'];order=['unet','deeplab','segformer'];records=[next(x for x in selection['scores'] if x['model_id']==k) for k in order]
fig,axes=plt.subplots(1,2,figsize=(8,3.5),sharey=True)
for ax,key,title in zip(axes,['validation_Dice','spatial_holdout_Dice'],['Validation (selection)','Spatial holdout, same position']):
 bars=ax.bar(labels,[r[key] for r in records],color=colors,width=.6);ax.set_ylim(0,1);ax.set_title(title,fontsize=10);ax.tick_params(axis='x',labelsize=9)
 for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,f'{bar.get_height():.3f}',ha='center',fontsize=9)
axes[0].set_ylabel('Root segmentation Dice');fig.tight_layout();fig.savefig(root/'three_model_comparison.png',dpi=600);fig.savefig(root/'three_model_comparison.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(6,3.5));x=np.arange(3);v=temporal['val']['per_class'];t=temporal['test']['per_class']
ax.bar(x-.18,[r['Dice'] or 0 for r in v],width=.36,color='#467565',label='Validation');ax.bar(x+.18,[r['Dice'] or 0 for r in t],width=.36,color='#746b91',label='Spatial holdout')
for i,r in enumerate(t):
 if r['Dice'] is None:ax.text(i+.18,.03,'NA',ha='center')
ax.set_xticks(x,['Persistent','Newly observed','Elongating']);ax.set_ylim(0,1);ax.set_ylabel('Temporal-class Dice');ax.legend(frameon=False);fig.tight_layout();fig.savefig(root/'temporal_class_metrics.png',dpi=600);fig.savefig(root/'temporal_class_metrics.pdf');plt.close(fig)
lines=['0353-2-B3：三模型根分割比较与四期预测','',
 '本次进度JSON原样保留。T1/T2标注与上一轮完全相同，复用已完成U-Net和上传TF1.12 DeepLabV3+训练，新增SegFormer-B0试训。',
 '只使用上部2541×1108像素区域；下半部分和分隔带全部排除。根分割图块256×256，训练60、验证20、空间留出20，同区T1/T2始终同组，跨组最少64像素间隔。','']
for r in records:lines.append(f"{r['model_id']}：验证Dice={r['validation_Dice']:.4f}，空间留出Dice={r['spatial_holdout_Dice']:.4f}。")
lines.extend(['','按验证Dice选择U-Net。此比较只针对根／背景分割，不是三模型时序类别排名。',
 '随后为选中架构训练独立时序分类：前后期RGB及冻结模型的两期根概率，共8通道；只有T1→T2人工标注进入训练。',
 f"时序分类验证三类平均Dice={temporal['val']['macro_Dice_supported_classes']:.4f}；留出平均仅涵盖有标注的{temporal['test']['classes_with_ground_truth']}类。"])
for r in temporal['val']['per_class']:lines.append(f"验证 {r['name']}：Dice={r['Dice']:.4f}，人工标注像素={r['support_pixels']}。")
lines.extend(['留出区无延伸根标注，该类留出Recall/Dice不可评估。',
 '四期结果均为模型预测。T1全部根单列为基础根；T2/T3/T4状态相对上一期判断。持续根不等于全部起源于T1。',
 'T3/T4完全按模型预测；原进度中的T3/T4笔迹不使用、不合并。T3/T4均未作为训练背景，时序类别准确率未评估，不能当作已验证根系追踪。',
 '低分类置信度/小类别差距保留为未判定；原始类别预测另存，不把所有根强行归类。颜色不代表死亡，未建立单根身份ID。',
 'editable_T1T2_manual_T3T4_model.json可导入自选图片版annotate.html：先选择原来四张完整RGB，再导入；T1/T2完全保留，T3/T4全部替换为模型预测，不自动确认背景，仍需复核。',
 'original_manual_label_review直接还原用户原始标注：T1蓝色基础根；T2紫色持续、绿色新生、黄色延伸。原标注与预测分开保存，原始temporal编码1新生、2延伸、3持续，无损保留。T2上部各类标注像素：持续69201、新生37360、延伸5824；这些都是原人工标签，不是模型生成标签。',
 '只有一个位置、两个已复核时期。报告的是与现有人工掩膜的一致度，非独立位置泛化；原标注粗细、漏标和遮挡会影响成绩。',
 '完整模型、脚本、来源校验和图块清单保留。DeepLabV3+原始模型权重在上一轮独立压缩包中，未冒充新训练。'])
(root/'结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig');print(json.dumps({'checks':len(checks),'temporal_validation_macro_Dice':temporal['val']['macro_Dice_supported_classes']}))
