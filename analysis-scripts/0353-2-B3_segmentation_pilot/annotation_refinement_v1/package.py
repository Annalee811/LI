"""Versioned ZIP delivery, each below GitHub's normal file-size limit."""
import argparse,shutil,zipfile
from pathlib import Path
import common as c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--publish',type=Path,required=True)
    ap.add_argument('--vendor',type=Path,required=True);ap.add_argument('--runtime-source',type=Path,required=True)
    ap.add_argument('--annotations-only',action='store_true');a=ap.parse_args()
    run=a.run;view=run/'revised_report';pub=a.publish;pub.mkdir(parents=True,exist_ok=True);source=Path(__file__).parent
    if not a.annotations_only:assert c.read(run/'tables/refinement_verification.json')['passed']
    prefix='0353-2-B3_annotation_refinement_v1' if a.annotations_only else '0353-2-B3_final'
    root=Path(prefix);base=Path('results/B3_annotation_refinement_v1')
    srcarc=Path('analysis-scripts/0353-2-B3_segmentation_pilot/annotation_refinement_v1');files=[]
    identity_assets=[(source.parents[1]/'dataset_identity.py',Path('analysis-scripts/dataset_identity.py'))]
    identity_assets += [(source.parents[2]/'dataset/current'/name,Path('dataset/current')/name)
        for name in ['identity_policy.json','README.md','active_positions.csv','active_images.csv','dataset_summary.json']]
    def write(name,items):
        f=pub/(prefix+'_'+name+'.zip')
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            seen=set()
            for src,arc in items:
                src=Path(src)
                ff=sorted(src.rglob('*')) if src.is_dir() else [src]
                for q in ff:
                    if not q.is_file() or '__pycache__' in q.parts:continue
                    target=root/arc/q.relative_to(src) if src.is_dir() else root/arc
                    if str(target) not in seen:z.write(q,target);seen.add(str(target))
        assert f.stat().st_size<99*1024*1024,('Package too large',f,f.stat().st_size)
        with zipfile.ZipFile(f) as z:
            assert z.testzip() is None
            assert not any(s.lower().endswith(('.doc','.docx')) for s in z.namelist())
        files.append(dict(file=f.name,bytes=f.stat().st_size,sha256=c.sha(f)))
        print('PACKAGE',f.name,f.stat().st_size,flush=True)
    if a.annotations_only:
        write('revised_annotations_RGB',[(run/'revision',base/'revision'),(run/'data/verification.json',Path('reference_verification.json')),
            (source/'REVISION_README.md',Path('README.md'))])
        c.js(pub/'annotation_review_SHA256.json',files);return
    # Quantitative and raster-rich figures are split to retain ordinary downloads.
    paper=[]
    for f in sorted((view/'figures').iterdir()):
        if f.is_file() and not f.name.startswith('All_test_'):
            paper.append((f,Path('figures')/f.name))
    write('paper_figures',paper+[(view/'tables',Path('tables')),(view/'manuscript',Path('manuscript'))])
    comp=view/'comparison_RGB'
    rgb=[(f,Path('RGB_comparison')/f.name) for f in comp.glob('*_RGB_annotated_comparison.png') if 'full_upper' not in f.name]
    rgb += [(comp/'native_local',Path('RGB_comparison/native_local')),(comp/'native_upper_reference',Path('RGB_comparison/native_upper_reference'))]
    write('four_period_RGB_comparison',rgb)
    write('full_upper_RGB_comparison',[(f,Path('full_RGB_comparisons')/f.name) for f in comp.glob('*_full_upper_RGB_annotated_comparison.png')])
    removed=[(comp/'background_removed_local',Path('background_removed/local')),(comp/'background_removed_upper',Path('background_removed/upper'))]
    removed += [(f,Path('background_removed')/f.name) for f in comp.glob('*background_removed*.png')]
    write('background_removed_full_local',removed)
    write('PlantPhenomics_guide',[(view/'manuscript',Path('manuscript')),(view/'tables',Path('tables')),
        (view/'figures/captions.json',Path('Figure_captions_EN.json')),(source/'README.md',Path('README.md'))])
    write('training_tiles_annotations_RGB',identity_assets+[(run/'data',base/'data'),(run/'revision/tile_masks',base/'revision/tile_masks'),
        (run/'revision/provenance.json',base/'revision/provenance.json'),(run/'revision/revision_log.csv',base/'revision/revision_log.csv'),
        (run/'starting_reference_snapshot',base/'starting_reference_snapshot'),(source,srcarc)] +
        [(f,base/'revision/native'/f.name) for f in (run/'revision/native').glob('*_revised_mask.png')])
    for model in c.MODELS:
        write('evaluation_'+model,[(run/model/'upper',base/model/'upper'),(run/model/'probabilities',base/model/'probabilities'),
            (view/'native_predictions'/model,Path('predictions')/model),(view/'tables',Path('tables')),
            (run/model/'selection.json',base/model/'selection.json'),(run/model/'metrics.json',base/model/'metrics.json'),
            (view/model/'metrics.json',base/'revised_report'/model/'metrics.json'),
            (view/'report_provenance.json',base/'revised_report/report_provenance.json')])
    torchitems=identity_assets+[(source,srcarc),(run/'torch_checkpoint_reload_verification.json',base/'torch_checkpoint_reload_verification.json')]
    for model in ['unet','segformer']:
        for name in ['best.pt','config.json','training_run.json','selection.json','history.csv','training_draws.csv','validation_thresholds.csv','baseline_validation.json']:
            torchitems.append((run/model/name,base/model/name))
    write('models_torch_scripts',torchitems)
    items=identity_assets+[(source,srcarc),(a.vendor,Path('results/B3_uploaded_deeplab_v1/vendor')),
        (a.runtime_source/'prepare_deeplab_runtime.py',Path('runtime/prepare_deeplab_runtime.py')),
        (a.runtime_source.parent/'Dockerfile',Path('runtime/Dockerfile')),
        (run/'tensorflow_checkpoint_reload_verification.json',base/'tensorflow_checkpoint_reload_verification.json')]
    for name in ['best_weights_fp16.npz','weight_names.json','config.json','training_run.json','selection.json','history.csv','training_draws.csv','validation_thresholds.csv','baseline_validation.json','deeplab_model_adapted.py']:
        items.append((run/'deeplab'/name,base/'deeplab'/name))
    write('model_deeplab_scripts',items)
    c.js(pub/'SHA256.json',files)
    for name in ['aggregate_metrics.csv','per_period_metrics.csv','root_length_agreement.csv','summary.json','original_holdout_metrics.csv']:
        shutil.copyfile(view/'tables'/name,pub/name)
    shutil.copyfile(run/'tables/refinement_verification.json',pub/'refinement_verification.json')
    for name in ['provenance.json','revision_log.csv']:shutil.copyfile(run/'revision'/name,pub/('revision_'+name))
    shutil.copyfile(run/'data/manifest.csv',pub/'random_tile_manifest.csv')
    title=['# 0353-2-B3 修订参考与本轮最终三模型结果','',
        '根系/背景任务，仅四期已标注上半部分。原始人工标注完整保留；修订版参考旧U-Net与RGB管状结构支持，新增根宽和漏标记录可逐像素追溯。它不是独立人工真值。',
        '200训练、20验证、24内部测试切块；25个训练基础位置各增加一个12像素平移窗口。四期同组，新增窗口排除验证/测试像素。训练窗之间有重叠，不是200张独立原始图。模型与阈值按原始验证标签选择；本交付图表统一对修订参考计算一致性。原始标签与原始保留测试指标另存供核查，不混入旧模型图。修订参考并非独立人工真值。',
        'U-Net、原上传TensorFlow DeepLabV3+及SegFormer均从对应旧权重微调20轮；若原始验证不改善，可以保留第0轮原权重。未修改原始RGB颜色或尺寸。','',
        '|模型|Precision|Recall|Dice|IoU|','|---|---:|---:|---:|---:|']
    for r in c.read(view/'tables/summary.json')['aggregate']:
        if r['split']=='test':title.append('|'+c.NAMES[r['model']]+'|'+'|'.join('%.4f'%r[k] for k in ['Precision','Recall','Dice','IoU'])+'|')
    title+=['','## ZIP下载','']+[f"- [{r['file']}]({r['file']}) ({r['bytes']/1024/1024:.1f} MiB)" for r in files]
    title+=['','表中是相对修订参考的一致性，不是独立人工准确率。主图Fig2三模型指标、Fig3四期带背景对比、Fig4三模型根长回归；S1训练曲线、S2逐期Dice、S3根长偏差、S4图块差异、S5误差定位。S5绿色TP、粉色FP、橙色FN均相对修订参考。根长由修订掩膜骨架推导，单位像素，不是独立人工尺量。外部独立测试图像数为0。',
        'ZIP包含CRC和SHA256校验，完整脚本、可编辑矢量统计图及600dpiTIFF。没有Word文件。公开下载的逐字节验证见public_download_verification.json。']
    (pub/'README.md').write_text('\n'.join(title)+'\n')

if __name__=='__main__':main()
