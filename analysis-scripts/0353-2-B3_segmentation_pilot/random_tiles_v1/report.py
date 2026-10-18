"""Honest manuscript text and Plant Phenomics submission guidance for this pilot."""
import argparse
from pathlib import Path
import numpy as np
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'manuscript';out.mkdir(parents=True,exist_ok=True)
    s=c.read(a.run/'tables/summary.json');agg=s['aggregate'];best=s['model_selected_by_validation_only'];periods=c.csvread(a.run/'tables/per_period_metrics.csv')
    test=[r for r in agg if r['split']=='test'];val=[r for r in agg if r['split']=='val'];lens=s['root_length']
    scores='; '.join(f"{r['name']}: precision {r['Precision']:.4f}, recall {r['Recall']:.4f}, Dice {r['Dice']:.4f}, IoU {r['IoU']:.4f}" for r in test)
    lengths='; '.join(f"{c.NAMES[r['model']]}: identity R²={r['R2_identity']:.4f}, RMSE={r['RMSE_px']:.2f} pixels, CCC={r['CCC']:.4f}" for r in lens)
    methods=("Four registered RGB images (T1: seedling; T2: heading; T3: maturity; T4: yellowing/senescence) from one annotated minirhizotron observation position (0353-2-B3) were used. Only the manually annotated upper field (2541 × 1108 registered-image pixels) was included. "
        "The field was directly sampled as 36 nonoverlapping 256 × 256-pixel tiles, without C0–C4 macro-region assignment or root-density/difficulty filtering. A single fixed random permutation (NumPy default_rng, seed 20261018) assigned 25 tile positions to training, 5 to validation and 6 to internal testing. "
        "All four dates at a given tile position remained in the same partition, yielding 100 training, 20 validation and 24 test RGB patches. These are correlated crops from 4 original images, not 144 independent acquisitions. Small geometric gaps prevented cross-partition pixel overlap; they do not establish biological or root-level independence. "
        "Original RGB pixels and supplied human masks were unchanged. Root = 255, background = 0 and uncertain = 128 (excluded). Unpainted pixels in the accepted upper field were treated as background. Augmentation was applied after splitting, on training tiles only.")
    training=("U-Net (ResNet-18 encoder and native RGB skip), the user-supplied original TensorFlow 1.12 DeepLabV3+ (ResNet-v2-50), and SegFormer-B0 (native-resolution RGB refinement head) were trained from random initialization. No previously B3-trained or ImageNet checkpoints were used. "
        "RGB encoder inputs were normalized as (RGB/255 − 0.5)/0.25; native RGB skip/refinement branches received RGB/255. Each epoch used 64 augmented draws from the 100 training images, with a shuffled cycling queue ensuring all training images were visited. The maximum was 50 epochs, batch size 2 and initial learning rate 0.0003 with cosine decay. "
        "The objective combined hard-negative binary cross-entropy, soft Dice and 0.15 soft clDice. Checkpoints were selected by validation Dice at threshold 0.5, checked every 2 epochs; probability thresholds were subsequently selected on validation only. "
        "The test set was excluded from model/threshold selection. PyTorch used CPU BF16 autocast for training and FP32 evaluation; the original TensorFlow implementation used FP32 and Adam with explicit L2 regularization, whereas PyTorch used AdamW. Thus this is a practical implementation comparison, not a perfectly controlled framework/optimizer ablation. CUDA was detected automatically and was unavailable in this run. "
        "No morphological bridging, dilation, component deletion or test-selected postprocessing was applied. This experiment trained root/background segmentation only, not persistent/new/extension temporal labels or death.")
    evaluation=("Foreground precision, recall, Dice and IoU were computed by pooling TP, FP and FN over the 24 test tile-date masks, counting each valid pixel once. Validation and test scores were reported separately and were not averaged with the earlier C3/C4 experiment. "
        "Per-date and four-date-pooled per-tile results are provided to expose heterogeneity. Undefined individual-tile ratios are null rather than assigned a perfect score; all tiles remain in pooled counts. "
        "Root lengths were derived by skeletonizing the human masks and predictions with identical graph-edge weighting (axial edges: 1 px; diagonal edges: √2 px; avoiding triangular double-counting). Agreement used identity R² = 1 − Σ(pred − human)²/Σ(human − mean(human))², RMSE, CCC and descriptive Bland–Altman plots. "
        "Lengths were not independently measured manually, and no physical pixel scale was available. The 24 tile-date pairs include repeated dates at 6 tile positions; they are not independent biological replicates. OLS R² is supplied separately and must not be substituted for identity R².")
    results=(f"On the random-tile internal test, the three models achieved {scores}. Model selection based solely on validation Dice identified {c.NAMES[best]}. "
        f"Mask-derived root-length agreement was {lengths}. "
        "These measured values should be reported as observed; near-perfect accuracy is not established. Whole-upper-field predictions include training pixels and are qualitative outputs, not independent test performance. Spatially accurate segmentation and accurate total root length answer different questions; high R² alone does not demonstrate correct fine-root boundaries or temporal tracking.")
    limits=("This is a single-position feasibility experiment. Random spatial tiling improves mixing of the sampled backgrounds but may produce optimistic performance because adjacent tiles can contain portions of the same roots and the four dates remain correlated. "
        "The new test excludes these tile pixels from this run's fitting/selection, but the image collection and earlier experiments have been repeatedly inspected. No claim of independent external or cross-tube validation is made. Human brush masks may not precisely trace anatomical root widths. "
        "Generalization to other positions/tubes must be evaluated separately; those predictions must not be described as quantified external accuracy without held-out human references. No biological confidence intervals, significance claims or fabricated measurements are supplied.")
    table=(['Model','Precision','Recall','Dice','IoU','Identity R²','RMSE(px)'],[[r['name']]+[f'{r[k]:.4f}' for k in ['Precision','Recall','Dice','IoU']]+[f"{next(x for x in lens if x['model']==r['model'])['R2_identity']:.4f}",f"{next(x for x in lens if x['model']==r['model'])['RMSE_px']:.2f}"] for r in test])
    cn=[('本轮划分与之前有什么区别','先在整幅上半部分直接切256×256原像素小图，再随机分配小图。没有先把大区C0–C4分成训练区/验证区/测试区。36个位置各自有T1–T4四张：25组=100张训练、5组=20张验证、6组=24张内部测试。同一位置四期始终同组，背景复杂或根少的小图没有被剔除。原始来源仍是一个位置的4张图，不能写成144张独立采集图。'),
        ('论文主文建议放哪些图','Fig1随机小图分配图，说明四期同坐标同组；Fig2三模型测试集Precision/Recall/Dice/IoU，作为主要定量结果；Fig3固定测试小图四期RGB背景上的人工标注与三模型标记；Fig4三模型根长一致性回归图；Fig5同一局部去背景、保留根原色的四期对比。若主文篇幅有限，Fig1和Fig5可移到补充材料。'),
        ('补充材料建议','FigS1训练曲线、FigS2逐期Dice、FigS3 Bland–Altman、FigS4所有测试小图的配对Dice、FigS5带RGB背景的误差定位图、FigS6黑根白底局部对比。All_test_Pxx图保留全部六个测试小图的四期结果。另有完整上半幅及局部的去背景根原色和黑根白底图、独立PNG。CSV表提供逐图块、逐时期、汇总指标和根长配对数据。训练/验证/测试数量与原始来源数量分开报告。'),
        ('应该放C3、C4还是平均值','本轮主结果使用新的随机小图内部测试集，不再用C3/C4作本轮集合名称。验证集只用于选权重和阈值，另列补充表；不要把验证集和测试集合并平均后当成测试准确率。旧C3/C4结果可作为历史空间留出实验单独列在补充材料，不能与本轮平均。'),
        ('如何分析准确性','先报告严格逐像素Dice和IoU，再看Precision与Recall的取舍；逐期和逐小图结果用于说明差异。主表汇总先累计TP/FP/FN再计算，不直接平均不同样本量的Dice。可以另列四期均值±样本标准差，但需注明日期相关，只作描述。不要把背景像素占比很高导致的总体Accuracy当作根系识别效果。'),
        ('根长R²应该如何写','三模型均有一致性图。R²以identity R²=1−SSE/SST为主要一致性量，同时报告RMSE、CCC及偏差；OLS R²另列。参考根长来自人工掩膜骨架，不能写成另外独立完成的人工尺量。单位是原始像素，暂不能换算成毫米。根长R²较高而Dice较低并不矛盾：根的总量接近，不代表位置和边界全部正确。'),
        ('Plant Phenomics上传方式','将Methods_Results_EN.md中的方法、结果和限制整合进你的完整论文主文，而不是直接当作完整论文提交。Fig1–Fig4按投稿系统要求分别上传TIFF或PDF，图注合并在文稿或系统指定栏；不要上传截图。补充图、CSV明细和说明组成补充文件。模型权重、训练图块、原始标注与完整脚本放公共仓库，在Data and Code Availability中引用版本固定的提交和校验表。'),
        ('文件格式与投稿前核对','已提供600dpi TIFF/PNG，以及PDF；定量图另有可编辑文字SVG。600dpi是图件排版输出，不代表原始RGB信息被增强。本文未核验Plant Phenomics当前在线投稿细则，最终尺寸、文件类型、许可与数据存储要求以其实时作者指南和投稿系统为准。作者需补充真实采集设计、实验方法、作者贡献和数据许可。不要把本预实验声称为对其他根管的独立验证。')]
    (out/'Methods_Results_EN.md').write_text('\n\n'.join(h+'\n'+t for h,t in [('Dataset',methods),('Training',training),('Evaluation',evaluation),('Results',results),('Limitations',limits)])+'\n')
    (out/'论文使用说明_中文.md').write_text('\n\n'.join(h+'\n'+t for h,t in cn)+'\n')
    (out/'Data_Code_Availability_EN.md').write_text(
        'Data and code availability\n\nThe annotated upper-field RGB images, unchanged human root masks, original annotation-progress JSON, '
        'fixed random tile allocation, trained model weights, prediction masks, evaluation tables and complete analysis scripts are available '
        'in Annalee811/LI, registration-results branch, under roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/random-tiles-v1. '
        'The immutable data commit and full-download SHA256 verification are recorded in public_download_verification.json; '
        'the individual ZIP checksums are listed in SHA256.json. The experiment uses four registered images from one annotated position; '
        'no independent external-position or external-tube test set is included.\n\nRepository: https://github.com/Annalee811/LI\n'
        'Before submission, insert the immutable data commit from the verification file into the citation. If the journal requires a DOI-bearing '
        'archive, deposit this fixed version in an appropriate long-term repository and cite its actual assigned DOI. Data ownership and licensing '
        'must be confirmed by the authors; no license or DOI is invented here.\n')
    c.js(out/'submission_map.json',dict(journal='Plant Phenomics',main_figures=['Fig1_random_tile_split','Fig2_test_segmentation_metrics','Fig3_four_date_qualitative_comparison','Fig4_three_model_root_length_agreement','Fig5_background_removed_local_RGB'],
        supplementary=['FigS1_learning_curves','FigS2_period_specific_Dice','FigS3_Bland_Altman','FigS4_paired_test_tile_Dice','FigS5_RGB_error_overlays','FigS6_background_removed_local_masks'],
        official_current_author_guidelines_verified=False,formats=['600dpi TIFF','600dpi PNG','PDF','SVG for quantitative figures'],
        manuscript_status='Methods/results fragments and guidance, not a complete paper or journal submission'))
    print('REPORT_DONE',flush=True)

if __name__=='__main__':main()
