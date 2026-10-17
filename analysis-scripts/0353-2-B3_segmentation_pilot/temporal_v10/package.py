"""ZIP delivery with source hashes, complete inputs and GitHub-safe part sizes."""
import argparse,json,shutil,zipfile
from pathlib import Path
import common as c

LIMIT=99*1024*1024
def writezip(dest,pairs):
    pairs=sorted(pairs,key=lambda r:r[1]);assert len({n for _,n in pairs})==len(pairs)
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f,arc in pairs:z.write(f,arc)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    assert dest.stat().st_size<LIMIT,(dest,dest.stat().st_size)
    print('DELIVERY_ZIP',dest.name,dest.stat().st_size,flush=True)
    return dict(file=dest.name,bytes=dest.stat().st_size,sha256=c.sha(dest),members=len(pairs))
def tree(base,prefix):return [(f,str(Path(prefix)/f.relative_to(base))) for f in base.rglob('*') if f.is_file() and '__pycache__' not in f.parts]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--publish',type=Path,required=True);p.add_argument('--features-only',action='store_true');a=p.parse_args();run=a.run;dest=a.publish;dest.mkdir(parents=True,exist_ok=True);records=[];source=Path(__file__).resolve().parent;repo=source.parents[2]
    for n in c.MODELS:
        f=dest/f'0353-2-B3_temporal_v10_frozen_features_{n}.zip'
        if not f.exists():
            pairs=tree(a.roots/n/'probabilities',n+'/probabilities')
            pairs += [(a.roots/n/filename,n+'/'+filename) for filename in ['postprocessing_selection.json','link_classifier.json']]
            pairs += [(a.roots/n/'images'/f'{t}_root_mask.png',f'{n}/images/{t}_root_mask.png') for t in c.PERIODS]
            records.append(writezip(f,pairs))
        else:records.append(dict(file=f.name,bytes=f.stat().st_size,sha256=c.sha(f)))
    if a.features_only:
        c.js(run/'feature_zip_manifest.json',records);return
    commonfiles=[(source/'README.md','README_method_and_reproduce.md')]
    for f in ['all_metrics.json','temporal_metrics.csv','per_transition_metrics.csv','root_length_agreement.json','root_length_pairs.csv','selection.json','verification.json','standalone_verification.json']:
        shutil.copyfile(run/f,dest/f)
    for f in ['README_method_and_reproduce.md']:shutil.copyfile(source/'README.md',dest/f)
    records.append(writezip(dest/'0353-2-B3_temporal_v10_paper_figures.zip',tree(run/'figures','figures')+[(run/f,f) for f in ['all_metrics.json','temporal_metrics.csv','per_transition_metrics.csv','root_length_agreement.json','root_length_pairs.csv','selection.json']]+commonfiles))
    records.append(writezip(dest/'0353-2-B3_temporal_v10_training_data.zip',tree(run/'data','data')+commonfiles))
    for n in c.MODELS:
        fs=tree(run/n/'four_period_final_fit',n+'/four_period_final_fit')+tree(run/n/'evaluation',n+'/evaluation')
        fs += [(f,n+'/'+f.name) for f in (run/n).glob('*') if f.is_file() and f.suffix in ['.json','.csv']]
        records.append(writezip(dest/f'0353-2-B3_temporal_v10_results_{n}.zip',fs+commonfiles))
    sources=[]
    for name in ['temporal_v10','paper_v6','root_background_v5']:
        sources+=tree(source.parent/name,'source/'+name)
    sources=[(f,n) for f,n in sources if f.suffix in ['.py','.md','.txt','.json']]
    sources+=tree(repo/'results/B3_uploaded_deeplab_v1/vendor','vendor')
    sources=[(f,n) for f,n in sources if f.suffix not in ['.png','.jpg']]
    sources += [(repo/'results/B3_uploaded_deeplab_v1/Dockerfile','DeepLab_Dockerfile')]
    heads=[]
    for n in c.MODELS:
        heads += [(f,n+'/'+f.name) for f in (run/n).glob('*') if f.is_file() and f.suffix in ['.pt','.json','.csv']]
    torches=[]
    for n in ['unet','segformer']:
        torches += [(a.roots/n/f,n+'/'+f) for f in ['best.pt','postprocessing_selection.json','link_classifier.json']]
    records.append(writezip(dest/'0353-2-B3_temporal_v10_weights_and_scripts.zip',sources+heads+torches+commonfiles))
    tf=[(a.roots/'deeplab'/f,'deeplab/'+f) for f in ['best_weights_fp16.npz','weight_names.json','postprocessing_selection.json','link_classifier.json']]
    records.append(writezip(dest/'0353-2-B3_temporal_v10_frozen_DeepLab_weights.zip',tf+commonfiles))
    c.js(dest/'download_manifest.json',dict(archives=records,all_zip_crc_verified=True,source_annotation_sha256=c.sha(run/'data/original_human_progress.json'),external_independent_test_images=0))
    report=json.loads((run/'all_metrics.json').read_text());selection=json.loads((run/'selection.json').read_text())
    text='# B3 三模型根系时序分割 v10\n\n在各自v9根系/背景分割基础上，先剔除背景，再训练三个独立时序分类头；根背景主干冻结。T1基础根；T2/T3/T4分别相对前一期，分类为持续、新生、生长。死亡不纳入。\n\n人工标注未更改，仅上半部。原始4张图、3个相邻转换；训练216对重叠切块、验证72对、同位置重复基准72对，外部独立测试图0。完整展示采用另训的全上半部最终头，是拟合展示；C3/C4评估来自仅在C0–C2训练的评估头。\n\n|模型|C3持续Dice|C3新生Dice|C3生长Dice|C3宏Dice|C4宏Dice|\n|---|---:|---:|---:|---:|---:|\n'
    for n in c.MODELS:
        m=report[n]['validation']['end_to_end'];vals=[r['Dice'] for r in m['classes']];text+='|%s|%.4f|%.4f|%.4f|%.4f|%.4f|\n'%(c.MODEL_NAMES[n],*vals,m['macro_Dice'],report[n]['repeated_benchmark']['end_to_end']['macro_Dice'])
    text+='\n验证选优：**'+c.MODEL_NAMES[selection['winner']]+'**。选优只用C3；C3参与轮数和校准选择，C4为已反复查看的同位置基准，两者均不能代替跨根管独立测试。各类支持量、Precision/Recall/Dice/IoU、混淆计数、校准前后数据均保存。生长段指标不足时如实报告，不等同于已验证的高精度对象追踪。\n\n|下载ZIP|大小MB|\n|---|---:|\n'
    for r in records:text+='|[%s](%s)|%.1f|\n'%(r['file'],r['file'],r['bytes']/1e6)
    text+='\n方法、复现和解释见[README_method_and_reproduce.md](README_method_and_reproduce.md)。三个frozen_features包合并解压为ROOT_V9，数据包解压为RUN/data；它们是复现所需的冻结第一阶段概率与根掩膜。权重脚本包包含全部三个时序头、U-Net/SegFormer根网络和完整源码，原始DeepLab根权重另包提供。全部ZIP小于GitHub单文件限制，下载后可按download_manifest.json核对SHA256。\n'
    (dest/'README.md').write_text(text)
if __name__=='__main__':main()
