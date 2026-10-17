#!/usr/bin/env python3
"""Immutable annotation audit and four-category three-model ZIP delivery."""
import argparse,csv,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import protocol as p
from evaluate_three_v9 import NAMES,result
from continuity_v8 import proposals,read
from evaluate_continuity_v8 import agreement

def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
def length_table(values):
 s='\n|模型|验证 R² / RMSE（像素）|C4 R² / RMSE（像素）|\n|---|---|---|\n'
 for n,label in NAMES.items():
  a=values[n]['val']['after'];b=values[n]['test']['after'];s+='|%s|%.4f / %.2f|%.4f / %.2f|\n'%(label,a['R2_agreement'],a['RMSE_px'],b['R2_agreement'],b['RMSE_px'])
 return s+'\n'
def zipout(path,entries):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for source,dest in entries:
   if source.is_dir():
    for f in sorted(source.rglob('*')):
     if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(dest/f.relative_to(source)))
   else:z.write(source,str(dest))
 with zipfile.ZipFile(path) as z:assert z.testzip() is None
 assert path.stat().st_size<100*1024**2,(path,path.stat().st_size)
 return dict(file=path.name,bytes=path.stat().st_size,SHA256=p.sha(path),CRC_verified=True)

def prepare(args):
 run=args.run;data=p.Data(args.data);reports=read(run/'all_metrics.json');comparison=[];perperiod=[]
 for name in NAMES:
  history=list(csv.DictReader((run/name/'history.csv').open(encoding='utf-8-sig')));threshold=float(history[0]['threshold'])
  for split,col in [('val',3),('test',4)]:
   folder=(args.unet_previous/'evaluation/v6_start'/split) if name=='unet' else (args.previous/name/'predictions'/split/('C'+str(col)))
   total=np.zeros(4,np.int64);per={}
   for t in p.PERIODS:
    prob=np.load(folder/(t+'_probability.npy'));cc=p.counts(data.gt[t][:,p.COLS[col]:p.COLS[col]+448],prob>=threshold);total+=cc;per[t]=p.metrics(cc)
   initial=dict(root=p.metrics(total),per_period=per,threshold=threshold);reports[name][split]['initial']=initial
   if split=='val':assert abs(initial['root']['Dice']-float(history[0]['validation_Dice']))<1e-12
   for mode in ['initial','before','after']:
    sc=reports[name][split][mode];comparison.append(dict(model=name,partition=split,stage=mode,**sc['root']))
    for t,m in sc['per_period'].items():perperiod.append(dict(model=name,partition=split,stage=mode,period=t,**m))
 p.js(run/'all_metrics.json',reports);p.csvout(run/'optimization_comparison.csv',comparison);p.csvout(run/'per_period_metrics.csv',perperiod)
 # Raw saved scores and RGB must exactly reproduce every final formal mask.
 checks=[]
 for name in NAMES:
  modeldir=run/name;sel=read(modeldir/'postprocessing_selection.json');classifier=read(modeldir/'link_classifier.json');checkpoint=modeldir/('best_weights_fp16.npz' if name=='deeplab' else 'best.pt');training=read(modeldir/'training_run.json');assert p.sha(checkpoint)==training['selected_checkpoint_SHA256']
  for split,col in [('val',3),('test',4)]:
   total=np.zeros(4,np.int64)
   for t in p.PERIODS:
    probfolder=modeldir/'probabilities'/('val' if split=='val' else 'test/C4');pr=np.load(probfolder/(t+'_probability.npy'));rgb=data.rgb[t][:,p.COLS[col]:p.COLS[col]+448];_,cc=proposals(pr,rgb,sel['threshold']);final,added,removed,links=result(pr,rgb,sel['threshold'],cc,sel['configuration'],classifier)
    saved=np.array(Image.open(modeldir/'evaluation'/split/(t+'_after.png')))==255;assert np.array_equal(final,saved);assert np.array_equal(np.array(Image.open(modeldir/'evaluation'/split/(t+'_human.png'))),data.gt[t][:,p.COLS[col]:p.COLS[col]+448]);total+=p.counts(data.gt[t][:,p.COLS[col]:p.COLS[col]+448],saved);checks.append(dict(model=name,partition=split,period=t,exact_final_mask_reproduction=True))
   assert p.metrics(total)==reports[name][split]['after']['root']
 p.js(run/'delivery_verification.json',dict(passed=True,exact_saved_mask_reproductions=len(checks),checks=checks,metrics_recomputed=True,manual_progress_SHA256=p.sha(args.data/'original_human_progress.json'),annotation_changed=False))
 dataset=read(run/'data/dataset/summary.json');text='# 0353-2-B3：三模型细根、断裂与背景误检优化（v9）\n\n'
 text+='三模型均完成本轮实际微调和连续性优化；原人工标注不改动，不纳入死亡或时序类别。下表对照起始网络的原始预测与本轮最终流程，不等同于和此前每一版完整后处理流程比较。网络保留轮次为0表示微调没有超过起始网络，保留旧网络权重；本轮训练、连接器与全部验证记录另行完整保存。\n\n'
 text+='|模型|保留网络轮次|验证 Dice：起始网络 → 最终|C4 Dice：起始网络 → 最终|最终 C4 Precision / Recall / IoU|\n|---|---:|---|---|---|\n'
 for name,label in NAMES.items():
  m=reports[name];epoch=read(run/name/'selection.json')['epoch'];q=m['test']['after']['root'];text+='|%s|%d|%.4f → %.4f|%.4f → %.4f|%.4f / %.4f / %.4f|\n'%(label,epoch,m['val']['initial']['root']['Dice'],m['val']['after']['root']['Dice'],m['test']['initial']['root']['Dice'],q['Dice'],q['Precision'],q['Recall'],q['IoU'])
 text+='\n训练来源为1个位置、4期完整上半部RGB。三个模型共用同一256次裁剪抽样序列，保存%d个不同的时期/窗口组合及全部裁剪参数；增强与重叠小块不会增加独立生物样本数。每模型验证40块、C4评价40块；U-Net用384上下文，另两模型用256上下文。正式指标融合整列后每像素计数一次。四期同一空间列始终属于同一组：C0–C2训练、C3验证、C4反复使用的同位置评价，跨组64像素间隔。**独立外部测试图像数为0**。\n'%dataset['training_unique_period_windows']
 text+='\n|数据部分|时期×空间列区域图数|每模型图块数|\n|---|---:|---:|\n|训练|12|256次抽样，189个不同时期/坐标窗口|\n|验证|4|40|\n|反复使用的同位置C4评价|4|40|\n|独立外部测试|0|0|\n\n这些区域来自相同4张原始时期图，不能称为20张独立原图。\n'
 text+='\n根长表与三个模型的回归图均提供：R²=1−SSE/SST、RMSE、MAE、CCC和Bland–Altman。参考为人工掩膜骨架长度，单位原像素；并无独立人工测长或物理标尺。32个块/时期配对相互关联，不作显著性推断。\n\n'
 text+=length_table(read(run/'root_length_agreement.json'))
 text+='连接诊断中的1像素邻域只用于训练连接分类和参考锚点连通性；Precision、Recall、Dice、IoU始终使用原始逐像素真值，无容差、不扩张真值。预测后处理还验证了0/0.025/0.05的概率滞后恢复：只从高阈值根像素出发，沿相邻有弱概率支持的像素生长，允许整套规则无改进时退回原预测。此操作不改变原始概率或人工掩膜。细根权重按人工笔迹局部宽度构造，不代表实际解剖直径。土壤、水珠、杂物合为背景，不声称训练了这些背景子类。新增像素包含弱响应恢复与几何补连，噪点移除像素与真值的符合/不符合数量均公开，不能把所有新增像素解释为真实根。\n\n'
 text+='论文图为600dpi PNG/TIFF和PDF，定量图另有SVG；采用克制的色盲友好配色。四期整图含训练像素，仅作展示。论文格式与视觉改善不代表模型已经达到0.9或具有跨位置泛化能力；以完整实测表为准。\n'
 (run/'Results_CN.md').write_text(text);print('V9_DELIVERY_PREPARED',flush=True)

def package(args):
 run=args.run;out=args.output;out.mkdir(parents=True,exist_ok=True);base=Path('0353-2-B3_three_model_v9');src=Path(__file__).resolve().parent
 files=['Results_CN.md','all_metrics.json','metrics.csv','optimization_comparison.csv','per_period_metrics.csv','root_length_agreement.json','root_length_pairs.csv','whole_field_changes.json','verification.json','delivery_verification.json','frozen_selection.json','Figure_captions_EN.md','README_reproduce.md','centerline_display_verification.json','standalone_inference_verification.json']
 common=[(run/f,base/f) for f in files];source=[]
 for f in ['refine_three_v9.py','refine_tf_v9.py','evaluate_three_v9.py','data_three_v9.py','figures_three_v9.py','deliver_three_v9.py','predict_three_v9.py','predict_tf_v9.py','continuity_v8.py','evaluate_continuity_v8.py','deliver_continuity_v8.py','train.py','protocol.py','figures.py','requirements.txt']:source.append((src/f,base/'source/paper_v6'/f))
 for f in ['common.py','train_torch.py','train_tf.py','evaluate.py']:source.append((src.parent/'root_background_v5'/f,base/'source/root_background_v5'/f))
 for f in ['train_uploaded_upper.py','train_segformer_progress.py']:source.append((src.parent/f,base/'source'/f))
 metadata=[]
 for n in NAMES:
  for f in ['training_run.json','config.json','history.csv','selection.json','thresholds.csv','postprocessing_selection.json','postprocessing_validation_grid.csv','link_classifier.json','link_training_audit.csv','link_training_features.npz']:metadata.append((run/n/f,base/n/f))
 bundles=[('paper_figures',common+[(run/'figures',base/'figures')]),('training_data',common+source+[(run/'data',base/'data')])]
 for n in NAMES:
  entries=common+[(run/n/'evaluation',base/n/'evaluation'),(run/n/'images',base/n/'images')]+[(f,d) for f,d in metadata if n in d.parts]
  for relative in ['val','test','upper']:entries.append((run/n/'probabilities'/relative,base/n/'probabilities'/relative))
  bundles.append(('segmentation_'+n,entries))
 bundles += [('models_torch',common+source+metadata+[(run/n/'best.pt',base/n/'best.pt') for n in ['unet','segformer']]),
  ('model_deeplab',common+source+metadata+[(run/'deeplab/best_weights_fp16.npz',base/'deeplab/best_weights_fp16.npz'),(run/'deeplab/weight_names.json',base/'deeplab/weight_names.json'),(args.vendor,base/'vendor'),(args.vendor.parent/'Dockerfile',base/'runtime/Dockerfile'),(args.vendor.parent/'source/prepare_deeplab_runtime.py',base/'runtime/prepare_deeplab_runtime.py')])]
 manifest=[]
 for name,entries in bundles:
  item=zipout(out/('0353-2-B3_v9_'+name+'.zip'),entries);manifest.append(item);print('V9_ARCHIVE',item,flush=True)
 p.js(out/'SHA256.json',dict(archives=manifest,categories=4,physical_ZIPs=len(manifest),explanation='Three model-specific result ZIPs and two framework weight ZIPs keep every file below GitHub100MiB.',manual_progress_SHA256=p.sha(args.data/'original_human_progress.json')))
 copy(run/'Results_CN.md',out/'README.md');copy(run/'optimization_comparison.csv',out/'optimization_comparison.csv');copy(run/'delivery_verification.json',out/'verification.json');copy(run/'README_reproduce.md',out/'README_reproduce.md')

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--previous',type=Path);a.add_argument('--unet-previous',type=Path);a.add_argument('--vendor',type=Path);a.add_argument('--output',type=Path);a.add_argument('--package',action='store_true');args=a.parse_args();package(args) if args.package else prepare(args)
if __name__=='__main__':main()
