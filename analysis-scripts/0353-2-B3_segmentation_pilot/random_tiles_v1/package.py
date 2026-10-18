"""Portable ZIPs below GitHub's regular-file limit, plus native output/data provenance."""
import argparse,shutil,zipfile
from pathlib import Path
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--publish',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True);p.add_argument('--runtime-source',type=Path,required=True);a=p.parse_args()
    assert c.read(a.run/'tables/verification.json')['passed']
    a.publish.mkdir(parents=True,exist_ok=True);prefix='0353-2-B3_random_tiles_v1';root=Path(prefix);source=Path(__file__).parent;files=[]
    def write(name,items):
        path=a.publish/(prefix+'_'+name+'.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for src,arc in sorted(items,key=lambda r:str(r[1])):
                src=Path(src)
                if src.is_dir():
                    for f in sorted(src.rglob('*')):
                        if f.is_file() and '__pycache__' not in f.parts:z.write(f,root/arc/f.relative_to(src))
                else:z.write(src,root/arc)
        assert path.stat().st_size<99*1024*1024,('ZIP too large for ordinary GitHub file',path)
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            assert not any(n.lower().endswith(('.doc','.docx')) for n in z.namelist()),'User requested no Word files'
        files.append(dict(file=path.name,bytes=path.stat().st_size,sha256=c.sha(path)))
        print('PACKAGE',path.name,path.stat().st_size,flush=True)
    srcarc=Path('analysis-scripts/0353-2-B3_segmentation_pilot/random_tiles_v1')
    basearc=Path('results/B3_random_tiles_v1')
    paper=[(f,Path('figures')/f.name) for f in (a.run/'figures').iterdir() if f.is_file() and not f.name.startswith('All_test_')]
    write('paper_figures',paper+[(a.run/'tables',Path('tables')),(a.run/'manuscript',Path('manuscript'))])
    comp=a.run/'comparison_RGB'
    comparisons=[(comp/'native_local',Path('RGB_comparisons/native_local')),(comp/'large_local',Path('RGB_comparisons/large_local')),(comp/'native_upper_human',Path('RGB_comparisons/native_upper_human')),(comp/'display_protocol.json',Path('RGB_comparisons/display_protocol.json'))]
    comparisons += [(f,Path('RGB_comparisons')/f.name) for f in comp.glob('*_RGB_annotated_comparison.png')]
    comparisons += [(f,Path('RGB_comparisons')/f.name) for f in (a.run/'figures').glob('All_test_*.png')]
    write('four_period_RGB_comparison',comparisons)
    removed=[(comp/'background_removed_local',Path('background_removed/local')),(comp/'background_removed_large_local',Path('background_removed/large_local')),(comp/'background_removed_upper',Path('background_removed/upper'))]
    removed.append((comp/'large_local/protocol.json',Path('background_removed/large_local/protocol.json')))
    removed += [(f,Path('background_removed')/f.name) for f in comp.glob('*background_removed*.png')]
    write('background_removed_full_and_local',removed)
    write('PlantPhenomics_guide',[(a.run/'manuscript',Path('manuscript')),(a.run/'tables',Path('tables')),(source/'README.md',Path('README.md'))])
    write('training_tiles_labels_RGB',[(a.run/'data',basearc/'data'),(source,srcarc),(a.run/'tables',Path('tables'))])
    for model in c.MODELS:
        items=[(a.run/'native_predictions'/model,Path('predictions')/model),(a.run/'data/upper',Path('original_upper_RGB_and_human_masks')),
            (a.run/model/'upper',basearc/model/'upper'),(a.run/model/'probabilities',basearc/model/'probabilities'),
            (a.run/'tables',Path('tables')),(a.run/model/'metrics.json',basearc/model/'metrics.json'),(a.run/model/'selection.json',basearc/model/'selection.json')]
        write('evaluation_'+model,items)
    items=[(source,srcarc)]
    for model in ['unet','segformer']:
        for name in ['best.pt','config.json','training_run.json','selection.json','history.csv','training_draws.csv','validation_thresholds.csv']:
            items.append((a.run/model/name,basearc/model/name))
    items.append((a.run/'torch_checkpoint_reload_verification.json',basearc/'torch_checkpoint_reload_verification.json'))
    items.append((a.run/'generic_native_inference_verification.json',basearc/'generic_native_inference_verification.json'))
    items.append((a.run/'training_draw_sequence_comparison.json',basearc/'training_draw_sequence_comparison.json'))
    write('models_torch_scripts',items)
    items=[(source,srcarc),(a.vendor,Path('results/B3_uploaded_deeplab_v1/vendor')),
        (a.runtime_source/'prepare_deeplab_runtime.py',Path('runtime/prepare_deeplab_runtime.py')),
        (a.runtime_source.parent/'Dockerfile',Path('runtime/Dockerfile'))]
    for name in ['best_weights_fp16.npz','weight_names.json','config.json','training_run.json','selection.json','history.csv','training_draws.csv','validation_thresholds.csv','deeplab_model_adapted.py']:
        items.append((a.run/'deeplab'/name,basearc/'deeplab'/name))
    items.append((a.run/'tensorflow_checkpoint_reload_verification.json',basearc/'tensorflow_checkpoint_reload_verification.json'))
    write('model_deeplab_scripts',items)
    c.js(a.publish/'SHA256.json',files)
    for name in ['aggregate_metrics.csv','per_period_metrics.csv','root_length_agreement.csv','summary.json','verification.json']:
        shutil.copyfile(a.run/'tables'/name,a.publish/name)
    shutil.copyfile(a.run/'data/manifest.csv',a.publish/'random_tile_manifest.csv');shutil.copyfile(a.run/'data/summary.json',a.publish/'dataset_summary.json')
    shutil.copyfile(a.run/'data/random_tile_allocation.png',a.publish/'random_tile_allocation.png')
    shutil.copyfile(a.run/'torch_checkpoint_reload_verification.json',a.publish/'torch_checkpoint_reload_verification.json')
    shutil.copyfile(a.run/'tensorflow_checkpoint_reload_verification.json',a.publish/'tensorflow_checkpoint_reload_verification.json')
    shutil.copyfile(a.run/'training_draw_sequence_comparison.json',a.publish/'training_draw_sequence_comparison.json')
    shutil.copyfile(a.run/'tables/visual_output_verification.json',a.publish/'visual_output_verification.json')
    header=['# 0353-2-B3：二维小图随机划分实验','',
        '只取四期已人工标注的上半部分；直接切256×256原像素小图，再固定随机分配。没有C0–C4大区分组。',
        '36个图块位置：25训练、5验证、6内部测试；四期对应100、20、24张RGB小图。同一坐标的四期始终同组，零跨集合像素重叠。',
        '三模型从零初始化，无旧B3或ImageNet权重。人工标签未修改，不按根系多少/难易筛图。验证集选权重和阈值，测试集仅评估。',
        '这是同一位置的内部预实验；外部独立位置/根管测试图像为0。新的随机小图测试不能与旧C3/C4数值混合平均。','',
        '|模型|Precision|Recall|Dice|IoU|','|---|---:|---:|---:|---:|']
    for r in c.read(a.run/'tables/summary.json')['aggregate']:
        if r['split']=='test':header.append('|'+r['name']+'|'+'|'.join('%.4f'%r[k] for k in ['Precision','Recall','Dice','IoU'])+'|')
    header+=['','## 下载','']+[f"- [{r['file']}]({r['file']}) ({r['bytes']/1024/1024:.1f} MiB)" for r in files]
    header+=['','论文主文建议Fig1–Fig4：随机划分、三模型测试指标、固定小图四期分割、三模型根长一致性；其他图与逐块明细作补充。英文方法、结果、图注及中文投稿说明见guide包。',
        '根长参考由人工掩膜骨架推导，单位为像素，并非独立人工尺量。原图整幅上半部分的分割包含训练像素，只作定性展示。',
        '所有ZIP已做CRC完整性验证，SHA256见SHA256.json；公开直链的逐字节校验结果另存于public_download_verification.json。']
    (a.publish/'README.md').write_text('\n'.join(header)+'\n')

if __name__=='__main__':main()
