#!/usr/bin/env python3
"""Audit exported pixels, write measured reports, and build downloadable archives."""
import argparse,csv,json,shutil,zipfile,platform,importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image
import protocol as p

FILES=['all_metrics.json','strict_ranking_metrics.json','segmentation_metrics.csv','noise_audit.csv',
 'root_length_pairs.csv','root_length_agreement.json','topology_boundary_metrics.json',
 'selection.json','delivery_selection.json','baseline_registry.json','stage_gate.json',
 'evaluation_verification.json','method_verification.json','export_integrity.json',
 'figure_style.json','figure_captions.json','Figure_captions_EN.md','Results_CN.md',
 'Methods_and_results_EN.md','environment.json','comparison_summary.csv']

def read(path):return json.loads(Path(path).read_text())
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)

def audit(data,run,selection):
 metrics=read(run/'all_metrics.json');checks=0
 for name in selection['candidates']:
  for split,col in [('val',3),('test',4)]:
   for mode,stem in [('raw','raw'),('cleaned','cleaned')]:
    counts=np.zeros(4,np.int64)
    for t in p.PERIODS:
     pred=np.array(Image.open(run/'candidates'/name/split/(t+'_'+stem+'.png')))==255
     gt=data.gt[t][:,p.COLS[col]:p.COLS[col]+448]
     counts+=p.counts(gt,pred);checks+=1
    assert p.metrics(counts)==metrics[name][split][mode]['root']
 for t in p.PERIODS:
  rgb=np.array(Image.open(run/'root_products'/(t+'_original_RGB.png')))
  gt=np.array(Image.open(run/'root_products'/(t+'_human_mask.png')))
  assert np.array_equal(rgb,data.rgb[t]) and np.array_equal(gt,data.gt[t]);checks+=2
  rgba=np.array(Image.open(run/'root_products'/(t+'_root_only_RGBA.png')))
  mask=np.array(Image.open(run/'root_products'/(t+'_cleaned_mask.png')))==255
  assert np.array_equal(rgba[:,:,3]>0,mask) and np.array_equal(rgba[mask,:3],rgb[mask]);checks+=1
 p.js(run/'export_integrity.json',dict(passed=True,checks=checks,
  all_reported_confusion_counts_reproduced_from_saved_masks=True,
  original_RGB_and_reference_masks_pixel_identical=True,
  root_only_RGBA_preserves_original_foreground_RGB=True,
  original_progress_SHA256=p.sha(data.source/'original_human_progress.json'),
  original_ranking_SHA256=p.sha(run/'selection.json'),delivery_selection_SHA256=p.sha(run/'delivery_selection.json')))

def report(run,selection):
 metrics=read(run/'all_metrics.json');winner=selection['winner'];length=read(run/'root_length_agreement.json')[winner]['test'];rows=[]
 for label,name,mode in [('Previous U-Net','unet','raw'),('Delivered U-Net ensemble',winner,'cleaned'),('Pretrained U-Net + TTA','refined_unet_tta','cleaned')]:
  for split in ['val','test']:rows.append(dict(configuration=label,split=split,**metrics[name][split][mode]['root']))
 p.csvout(run/'comparison_summary.csv',rows)
 table='|配置／数据范围|Precision|Recall|Dice|IoU|\n|---|---:|---:|---:|---:|\n'
 for r in rows[:4]:
  label=('原 U-Net' if r['configuration']=='Previous U-Net' else '本次交付')+'／'+('验证 C3' if r['split']=='val' else '重复留出 C4')
  table+='|'+label+'|'+'|'.join(f'{r[k]:.4f}' for k in ['Precision','Recall','Dice','IoU'])+'|\n'
 with (run/'noise_audit.csv').open(encoding='utf-8-sig') as f:noise=[r for r in csv.DictReader(f) if r['candidate']==winner]
 totals={s:{k:sum(int(r[k]) for r in noise if r['split']==s) for k in ['total_removed_pixels','removed_reference_background_pixels','removed_reference_root_pixels']} for s in ['val','test']}
 text=f'''# 0353-2-B3 根／背景分割与论文图 v6

本次交付四期上部原像素 RGB、原始人工掩膜、原始预测、保守去噪掩膜、透明背景根系图和叠加图；另附六幅主图、两幅补充图及全部实际评估表。颜色表示根／背景和误差，不表示新生、持续或延伸；死亡未纳入。

{table}
原 U-Net 为上一版已发布阈值的原始输出；本次交付为按 C3 验证选择的预训练 U-Net 与原 U-Net 等权集成，四方向翻转平均、384 像素上下文。没有修改人工真值，没有通过膨胀真值或加粗根系提高 Dice。新训练模型完成60轮，最佳权重第35轮；其单模型 TTA 的重复 C4 Dice 为 {metrics['refined_unet_tta']['test']['cleaned']['root']['Dice']:.4f}，完整保留，但没有根据 C4 重新更换交付模型。

## 去噪的实测影响

沿用已冻结的模型与0.575阈值，去除面积不超过20像素、短而紧凑、最高分低于0.775的孤立连通域。细长细根和高分小目标保留，不使用腐蚀或开运算。此参数从既有 C3 网格按 Precision 选取，允许 Dice 损失最多0.001，并限制 Recall 损失。原始严格排名、交付清理决策分别保存在 selection.json、delivery_selection.json。

四期 C3 共删除 {totals['val']['total_removed_pixels']} 像素，其中 {totals['val']['removed_reference_background_pixels']} 个属于参考背景、{totals['val']['removed_reference_root_pixels']} 个属于参考根；C4 共删除 {totals['test']['total_removed_pixels']} 像素，其中背景 {totals['test']['removed_reference_background_pixels']}、参考根 {totals['test']['removed_reference_root_pixels']}。去噪有细小的召回损失，因此原始预测与清理结果同时提供。不能把每个删除对象都认定为已确认的噪点。

## 样本与评估

只有1个位置、4张原始时期 RGB。各图只使用上部2541×1108像素，未缩放、未修改颜色。384×384原像素切块：训练120、验证40、重复空间留出40；对应30、10、10个物理窗口，每个窗口含四期，同一窗口四期始终同组。跨组64像素隔离；组内图块重叠，不能当成200个独立样本。正式指标对整列概率融合后逐像素计数一次。

C4 在前次实验和本轮中已被查看，是同一位置的重复探索性基准；独立外部测试图像数为0。交付清理是初次排名评估后从已计算的 C3 网格新增选择，C4 未用于此参数选择。模型训练轮数、预训练及推理策略不同，三模型结果比较不等同于等预算架构优劣结论。背景仅一个合并类别，没有土壤／水滴／杂物的独立人工真值。

根长一致性：重复 C4 的 R²＝{length['R2_agreement']:.4f}，RMSE＝{length['RMSE_px']:.2f}原始像素，CCC＝{length['CCC']:.4f}。参考长度由人工笔刷掩膜骨架计算，并非独立人工长度测量；无物理标尺。32个区块—时期对来自8个非重叠区块重复四期，存在相关性，不报告独立样本显著性。原始参考笔迹较细，指标反映与这些标注的一致性，不等同于完整根体边界的准确度。

## 图片

Fig1四期全景；Fig2相同固定空间窗口的原图／人工／预测／误差；Fig3模型指标；Fig4根长一致性与 Bland–Altman；Fig5 PR 与学习曲线；Fig6原像素去噪细节；FigS1分时期指标；FigS2根系得分。白底、灰色参考、青绿色根系，橙／蓝表示误差。主图输出600dpi PNG、LZW TIFF及PDF，定量图另有SVG。图片排版质量与模型准确率分别评价；600dpi输出不增加原始照片的信息。

所有原始人工进度完整保留。当前严格 Dice 尚未达到0.95的既定门槛，未进入时序分类或其他位置预测。后续提分仍需由真实未见数据验证，不能以后处理或图片美化替代验证。
'''
 (run/'Results_CN.md').write_text(text,encoding='utf-8')
 m=metrics[winner]['test']['cleaned']['root'];v=metrics[winner]['val']['cleaned']['root']
 en=f'''# Methods and measured results — exploratory single-position study

Four aligned upper RGB fields (2541×1108 native pixels) and unchanged user brush masks from 0353-2-B3 were used for binary root/background segmentation. Root temporal attributes were preserved but excluded from training; mortality was excluded. Training, validation and repeated benchmark columns were separated by64 pixels; all four dates of each spatial window shared a partition. The384×384 export contains120/40/40 patch images, equivalent to30/10/10 physical windows, not independent biological replicates. There were no external independent test images.

A ResNet18 ImageNet-initialized U-Net with a native RGB skip was trained using hard-negative binary cross-entropy, soft Dice and0.15 soft clDice. The encoder was frozen for5 epochs; pretrained batch-normalization running statistics stayed frozen. AdamW used encoder/decoder learning rates2e-5/3e-4, weight decay1e-4, batch2 and48 dynamic training crops per epoch. RGB flips, right-angle rotations and brightness/gamma augmentation were restricted to training. Training stopped at60 epochs; the checkpoint with the highest validation Dice at threshold0.5 was epoch35. CUDA was detected automatically; this run used CPU. Prediction used native384-pixel overlapping windows and Hann fusion, with no RGB from outside the evaluation column.

Validation-only exploration included the previous three configurations, larger context, flip averaging, the pretrained U-Net and fixed equal-weight ensembles. The frozen model winner was the equal mean of the old and pretrained U-Nets with four-flip averaging. After initial ranking, a conservative delivery filter was chosen from the precomputed validation grid, without using C4 scores. At the unchanged0.575 threshold, compact components of area≤20, moment elongation<2.5 and maximum score<0.775 were removed. This trades a small number of reference-root pixels for fewer background predictions. Both raw and filtered masks and the original selection are retained. All explored configurations and confusion counts are reported.

Delivered root Precision/Recall/Dice/IoU were {v['Precision']:.4f}/{v['Recall']:.4f}/{v['Dice']:.4f}/{v['IoU']:.4f} on C3 and {m['Precision']:.4f}/{m['Recall']:.4f}/{m['Dice']:.4f}/{m['IoU']:.4f} on C4. C4 had been inspected during previous experiments and was revisited here; these scores describe a repeated within-position benchmark, not a fresh independent test. They do not establish cross-position generalization or near-perfect accuracy. Optimization budgets differed between model configurations; this is not an equal-budget architecture comparison.

Native-pixel skeleton lengths were derived from both brush-reference and prediction masks using orthogonal/diagonal edge weights. Lengths were computed before block subdivision. Across32 correlated block–date pairs (8 spatial blocks×4 dates), agreement R²=1−SSE/SST was {length['R2_agreement']:.4f}, RMSE {length['RMSE_px']:.2f}px and CCC {length['CCC']:.4f}. No independent manual length measurements, physical calibration, or inferential significance claim is available. Strict foreground pixel metrics are primary; clDice and boundary F1 at1px tolerance are separate diagnostics. Native RGB and reference masks were not enhanced, rescaled, dilated or redrawn. Scores express agreement with the supplied brush annotations, not an independently adjudicated anatomical ground truth.
'''
 (run/'Methods_and_results_EN.md').write_text(en,encoding='utf-8')
 packages=['torch','torchvision','numpy','Pillow','scipy','scikit-image','transformers','matplotlib']
 p.js(run/'environment.json',dict(python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in packages},device='cpu',CUDA_available=False,training_threads=3,
  initial_weights=dict(url='https://download.pytorch.org/models/resnet18-f37072fd.pth',SHA256='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec')))

def archive(target,base,entries):
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for source,name in entries:
   if source.is_dir():
    for f in sorted(source.rglob('*')):
     if f.is_file() and '__pycache__' not in f.parts:z.write(f,str(base/ name/f.relative_to(source)))
   else:z.write(source,str(base/name))
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
 assert target.stat().st_size<100*1024**2,'GitHub per-file hard limit exceeded'
 return dict(file=target.name,bytes=target.stat().st_size,SHA256=p.sha(target),CRC_verified=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--run',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 data=p.Data(args.data);run=args.run;sel=read(run/'delivery_selection.json');audit(data,run,sel);report(run,sel)
 out=args.output;out.mkdir(parents=True,exist_ok=True);sources=Path(__file__).resolve().parent;helper=sources.parent;manifest=[]
 reports=[(run/f,Path('evaluation')/f) for f in FILES]
 for name in sel['candidates']:
  for f in ['validation_cleanup.json','validation_cleanup_grid.csv']:
   reports.append((run/'candidates'/name/f,Path('evaluation/validation_search')/name/f))
 manifest.append(archive(out/'0353-2-B3_paper_figures_v6.zip',Path('0353-2-B3_paper_figures_v6'),[(run/'paper_figures',Path('figures'))]+reports))
 entries=[(run/'root_products',Path('images'))]+reports+[(data.source/'original_human_progress.json',Path('original_human_progress.json'))]
 for split in ['val','test']:entries.append((run/'candidates'/sel['winner']/split,Path('isolated_benchmark')/split))
 manifest.append(archive(out/'0353-2-B3_root_results_v6.zip',Path('0353-2-B3_root_results_v6'),entries))
 # A portable model bundle, including frozen TensorFlow baseline scores for numerical comparisons.
 entries=[(run/'unet_pretrained',Path('unet_pretrained')),(sources,Path('source/paper_v6')),
  (helper/'root_background_v5',Path('source/root_background_v5')),
  (helper/'train_uploaded_upper.py',Path('source/train_uploaded_upper.py')),
  (helper/'train_segformer_progress.py',Path('source/train_segformer_progress.py'))]+reports
 for name in ['unet','segformer']:
  for f in ['best.pt','training_run.json']:entries.append((data.source/name/f,Path('legacy')/name/f))
 entries.append((data.source/'deeplab/training_run.json',Path('legacy/deeplab/training_run.json')))
 for split,col in [('val','C3'),('test','C4')]:
  for t in p.PERIODS:entries.append((data.source/'deeplab/predictions'/split/col/(t+'_probability.npy'),Path('legacy/deeplab/predictions')/split/col/(t+'_probability.npy')))
 # Selection and registry belong at run root for --run model_bundle.
 for f in ['selection.json','delivery_selection.json','baseline_registry.json']:entries.append((run/f,Path(f)))
 manifest.append(archive(out/'0353-2-B3_refined_models_v6.zip',Path('0353-2-B3_refined_models_v6'),entries))
 manifest.append(archive(out/'0353-2-B3_native384_training_data_v6.zip',Path('0353-2-B3_native384_training_data_v6'),[(run/'data_package',Path('.')),(sources/'README.md',Path('README.md'))]))
 p.js(out/'SHA256.json',dict(archives=manifest,original_progress_SHA256=p.sha(data.source/'original_human_progress.json'),RGB_and_ground_truth_unchanged=True))
 copy(run/'Results_CN.md',out/'README.md');copy(run/'comparison_summary.csv',out/'comparison_summary.csv')
 for item in manifest:print('VERIFIED_ARCHIVE',item,flush=True)
if __name__=='__main__':main()
