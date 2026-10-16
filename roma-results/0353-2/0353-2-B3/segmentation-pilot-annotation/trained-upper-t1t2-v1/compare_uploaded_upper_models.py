#!/usr/bin/env python3
"""Combine actual U-Net and uploaded TF1 DeepLab results with shared held-out masks."""
from pathlib import Path
import argparse,csv,hashlib,json,shutil,zipfile
import numpy as np
from PIL import Image,ImageDraw,ImageFont

p=argparse.ArgumentParser();p.add_argument('--unet',type=Path,required=True);p.add_argument('--deeplab',type=Path,required=True);a=p.parse_args();u=a.unet;d=a.deeplab
um=json.loads((u/'evaluation/metrics.json').read_text());dm=json.loads((d/'training/metrics.json').read_text());ur=json.loads((u/'training/training_run.json').read_text());dr=json.loads((d/'training/training_run.json').read_text());summary=json.loads((u/'dataset/dataset_summary.json').read_text())
assert ur['manifest_sha256']==dr['manifest_sha256']
assert hashlib.sha256((u/'dataset/manifest.csv').read_bytes()).hexdigest()==dr['manifest_sha256']
um['model_comparison_status']='This file reports U-Net only; uploaded TensorFlow DeepLabV3+ is reported in model_comparison.csv and deeplab/metrics.json. RootPainter has not been trained.'
(u/'evaluation/metrics.json').write_text(json.dumps(um,ensure_ascii=False,indent=2)+'\n')
readme=(u/'README.md').read_text().replace('未训练或比较 RootPainter、DeepLabV3+；','DeepLabV3+结果见model_comparison.csv及deeplab/；RootPainter尚未训练；')
readme+='\n上传模型的权重单独保存在DeepLabV3plus_TF112_model.zip，数据与结果包包含其指标及预测；两份ZIP可分别解压。\n'
(u/'README.md').write_text(readme)
checks=[];rows=[]
assert hashlib.sha256((d/'training/best_weights_fp16.npz').read_bytes()).hexdigest()==dr['checkpoint_sha256']
weight_names=json.loads((d/'training/weight_names.json').read_text())
with np.load(d/'training/best_weights_fp16.npz') as weights:
    assert len(weights.files)==len(weight_names)
    assert all(np.isfinite(weights[k]).all() and weights[k].dtype==np.float16 for k in weights.files)
    assert sum(weights[k].size for k in weights.files)==dr['parameters_including_batch_norm_state']
checks.append(dict(name='DeepLab checkpoint hash, variable count, dtype and finite values',passed=True))
original=(d/'vendor/deeplab_model.py').read_text();adapted=(d/'training/deeplab_model_adapted.py').read_text()
assert adapted==original.replace('    if is_training:\n      exclude =','    if is_training and pre_trained_model:\n      exclude =')
assert dr['backbone']=='resnet_v2_50' and dr['aspp_depth']==256 and dr['tensorflow_version']=='1.12.0'
assert not dr['test_used_for_training_or_selection']
checks.append(dict(name='Actual uploaded TF1.12 architecture with only documented loading guard',passed=True))
for split,key,col in [('val','validation',3),('test','spatial_holdout_test',4)]:
    for t in ['T1','T2']:
        ug=np.array(Image.open(u/f'predictions/{split}/{t}_C{col}_human_mask.png'))
        dg=np.array(Image.open(d/f'training/predictions/{split}/{t}_C{col}_human_mask.png'))
        assert np.array_equal(ug,dg)
        gt=dg==255;valid=dg!=128;pr=np.array(Image.open(d/f'training/predictions/{split}/{t}_C{col}_prediction.png'))==255
        c=np.array([(gt&pr&valid).sum(),(~gt&pr&valid).sum(),(gt&~pr&valid).sum(),(~gt&~pr&valid).sum()])
        checks.append(dict(name=f'{split}_{t}_identical_ground_truth',passed=True))
        if t=='T1':total=c
        else:total+=c
    m=dm[key]['micro_metrics'];assert all(int(m[k])==int(x) for k,x in zip(['TP','FP','FN','TN'],total))
    assert total.sum()==992768
    checks.append(dict(name=f'{split}_DeepLab_counts_reproduce_and_overlap_counted_once',passed=True))
    for name,metrics,run in [('U-Net',um,ur),('DeepLabV3+ (uploaded TF1.12 / ResNet-v2-50)',dm,dr)]:
        m=metrics[key]['micro_metrics'];rows.append(dict(model=name,split=split,Precision=m['Precision'],Recall=m['Recall'],Dice=m['Dice'],IoU=m['IoU'],threshold=run['frozen_threshold'],training_epochs=run['epochs_completed'],best_epoch=run['best_epoch'],n_tile_images=20,n_unique_evaluated_pixels=992768))
with (u/'model_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Export scientific plots of actual metrics; no pseudo replicates or invented uncertainty.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
keys=['Precision','Recall','Dice','IoU'];xx=np.arange(4)
for ax,split,label in zip(axes,['val','test'],['Validation (threshold tuned)','Spatial holdout (same position)']):
    rr=[r for r in rows if r['split']==split]
    for j,(r,color) in enumerate(zip(rr,['#467565','#746b91'])):
        bars=ax.bar(xx+(j-.5)*.35,[r[k] for k in keys],width=.35,color=color,label='U-Net' if j==0 else 'DeepLabV3+')
        for bar in bars:ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,f'{bar.get_height():.3f}',ha='center',fontsize=8)
    ax.set_xticks(xx,keys);ax.set_ylim(0,1);ax.set_title(label);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Agreement with supplied human mask');axes[1].legend(frameon=False,loc='upper right');fig.tight_layout()
fig.savefig(u/'model_comparison.png',dpi=600);fig.savefig(u/'model_comparison.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
for folder,label,color in [(u/'training','U-Net','#467565'),(d/'training','DeepLabV3+','#746b91')]:
    hist=list(csv.DictReader((folder/'history.csv').open(encoding='utf-8-sig')))
    ep=[int(r['epoch']) for r in hist]
    axes[0].plot(ep,[float(r['train_loss']) for r in hist],label=label,c=color)
    axes[1].plot(ep,[float(r['validation_Dice']) for r in hist],label=label,c=color)
axes[0].set(xlabel='Epoch',ylabel='Training loss');axes[1].set(xlabel='Epoch',ylabel='Validation Dice at threshold 0.5',ylim=(0,1));axes[1].legend(frameon=False)
fig.tight_layout();fig.savefig(u/'two_model_training_curves.png',dpi=300);fig.savefig(u/'two_model_training_curves.pdf');plt.close(fig)

# Preserve all original human annotations and dataset; include the second model's outputs alongside them.
target=u/'deeplab';target.mkdir(exist_ok=True)
for name in ['metrics.json','training_run.json','config.json','history.csv','validation_thresholds.csv','weight_names.json','best_epoch.json','per_column_metrics.csv','deeplab_model_adapted.py']:
    shutil.copyfile(d/'training'/name,target/name)
shutil.copyfile(d/'source_provenance.json',target/'source_provenance.json');shutil.copyfile(d/'vendor/LICENSE',target/'UPLOADED_MODEL_LICENSE.txt')
shutil.copytree(d/'training/predictions',target/'predictions',dirs_exist_ok=True)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',36)
for t in ['T1','T2']:
    rgb=np.array(Image.open(u/f'dataset/upper/{t}_RGB.png'));g=np.array(Image.open(u/f'dataset/upper/{t}_mask.png'))==255
    up=np.array(Image.open(u/f'predictions/{t}_upper_prediction_mask.png'))==255
    dp=np.array(Image.open(target/f'predictions/upper/{t}_upper_prediction_mask.png'))==255
    h,w=rgb.shape[:2];canvas=Image.new('RGB',(w*4,h+60),'white');draw=ImageDraw.Draw(canvas)
    for j,(sel,label,color) in enumerate([(None,t+' original RGB',None),(g,'User-supplied human mask',[0,170,230]),(up,'U-Net prediction',[25,180,100]),(dp,'Uploaded DeepLabV3+ prediction',[25,180,100])]):
        im=rgb.copy()
        if sel is not None:im[sel]=(im[sel]*.35+np.array(color)*.65).astype(np.uint8)
        canvas.paste(Image.fromarray(im),(j*w,60));draw.text((j*w+12,7),label,font=font,fill='#20352d')
    canvas.save(u/f'{t}_comparison.png',dpi=(300,300))
# No cherry-picking by score: fixed test views at y=0 and y=512 for both periods.
canvas=Image.new('RGB',(1024,1176),'white');draw=ImageDraw.Draw(canvas);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
for i,(t,y) in enumerate([(t,y) for t in ['T1','T2'] for y in [0,512]]):
    rgb=np.array(Image.open(u/f'dataset/upper/{t}_RGB.png'))[y:y+256,2070:2326]
    gt=np.array(Image.open(u/f'predictions/test/{t}_C4_human_mask.png'))[y:y+256,:256]
    up=np.array(Image.open(u/f'predictions/test/{t}_C4_prediction.png'))[y:y+256,:256]
    dp=np.array(Image.open(target/f'predictions/test/{t}_C4_prediction.png'))[y:y+256,:256]
    for j,(im,label) in enumerate([(rgb,'RGB'),(gt,'Human mask'),(up,'U-Net'),(dp,'DeepLabV3+')]):
        im=Image.fromarray(im).convert('RGB');canvas.paste(im,(j*256,i*294+38));draw.text((j*256+5,i*294+5),t+' '+label,font=small,fill='black')
canvas.save(u/'heldout_two_model_comparison.png',dpi=(300,300))
# Put human-readable actual scores first; document pilot limits and unequal model budgets.
lines=['0353-2-B3 上部 T1/T2：两模型分割预实验','',
       '人工标注直接使用，不需要再补标。只用上方2541×1108区域；下方与白带不参与。',
       '原分辨率256×256切块，共100张：训练60、验证20、空间留出测试20。',
       '50个物理观察窗，每窗T1/T2严格同组；跨组最小64像素间隔，原RGB和标注不缩放、不膨胀。',
       '重叠预测先融合，再统计每个唯一像素；每个评估组共992768个有效像素。','']
for r in rows:
    lines.append(f"{r['model']} | {r['split']}：Precision={r['Precision']:.4f}, Recall={r['Recall']:.4f}, Dice={r['Dice']:.4f}, IoU={r['IoU']:.4f}")
lines.extend(['',f"U-Net完成{ur['epochs_completed']}轮，最佳第{ur['best_epoch']}轮，阈值{ur['frozen_threshold']:.2f}。",f"上传的TF1.12 DeepLabV3+完成{dr['epochs_completed']}轮，最佳第{dr['best_epoch']}轮，阈值{dr['frozen_threshold']:.2f}。",
              '两模型均随机初始化，未使用预训练权重。数据和测试区域相同，但模型大小、优化器、训练轮数不同，因此属于首轮预实验比较，不是严格控制训练预算的最终排名。',
              'DeepLab使用上传仓库原始生成器：ResNet-v2-50 + 256通道ASPP + 原始解码器。仅使缺少预训练权重时能跳过加载。',
              'DeepLab权重在单独的压缩包中；以FP16存储、恢复到FP32图推理，指标均在重新载入该权重后计算。','',
              '指标描述与现有人工掩膜的一致性；未标细根和部分中心线式标注会影响训练、指标与根宽。',
              '仅一个微根管位置、两个时期，空间留出不等于独立地点/生物学独立测试；同一根仍可能跨过分组。',
              '整幅预测包含训练区域，仅用于观察，报告的测试成绩只来自留出列。',
              '新生、延伸、持续等人工属性已保留，但本轮仅训练根／背景，不自动判断死亡。',
              '未训练RootPainter；没有独立人工根长测量或像素物理标尺，不生成其R²/RMSE。'])
(u/'结果说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
comparison=dict(models=['U-Net','Uploaded TensorFlow 1.12 DeepLabV3+'],scores=rows,shared_manifest_sha256=ur['manifest_sha256'],same_test_masks=True,test_independence='within-position spatial holdout only',ground_truth_limitations=summary['limitations'],deep_checkpoint_reloaded_before_scoring=True)
(u/'model_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
(u/'two_model_verification.json').write_text(json.dumps(dict(passed=len(checks),failed=0,checks=checks),indent=2)+'\n')
print(json.dumps({'comparison':rows,'passed_checks':len(checks)}))
