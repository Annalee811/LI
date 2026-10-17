#!/usr/bin/env python3
"""Export reviewable result/data/model ZIPs, all under GitHub's100MiB limit."""
import argparse,json,zipfile
from pathlib import Path
import common as c

def files(root):return sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
def build(path,entries):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for source,rel in entries:z.write(source,rel)
    assert path.stat().st_size<100*1024**2,'Split artifact before GitHub upload'
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    return dict(filename=path.name,bytes=path.stat().st_size,sha256=c.sha(path),files=len(entries))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['data','all'],default='all');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    repo=Path.cwd().resolve();src=Path(__file__).resolve().parent;records=[]
    vendor=repo/'results/B3_uploaded_deeplab_v1/vendor'
    if not vendor.exists():vendor=repo/'vendor'
    prefix='0353-2-B3_root_background_v5'
    dataset=[(x,'dataset/'+str(x.relative_to(a.run/'dataset'))) for x in files(a.run/'dataset')]
    for name in ['original_human_progress.json','annotation_audit.json','verification.json','input_verification.json','metric_verification.json','environment.json','spatial_split.png']:
        path=a.run/name
        if path.exists():dataset.append((path,name))
    dataset += [(x,'source/root_background_v5/'+x.name) for x in files(src)]
    records.append(build(a.output/(prefix+'_data.zip'),dataset))
    if a.mode=='all':
        assert (a.run/'stage_gate.json').exists() and (a.run/'evaluation_verification.json').exists()
        entries=[]
        for name in ['结果说明.md','selection.json','stage_gate.json','all_metrics.json','segmentation_metrics.csv',
            'root_length_pairs.csv','root_length_agreement.json','topology_metrics.json','evaluation_verification.json','input_verification.json',
            'annotation_audit.json','label_geometry_audit.json','metric_verification.json','environment.json','unet_optimization.json','spatial_split.png']:
            entries.append((a.run/name,name))
        entries += [(a.run/'dataset/summary.json','dataset_summary.json')]
        for folder in ['figures','selected_root_products']:
            entries += [(x,folder+'/'+str(x.relative_to(a.run/folder))) for x in files(a.run/folder)]
        for model in ['unet','deeplab','segformer']:
            entries += [(x,'models/'+model+'/'+x.name) for x in (a.run/model).iterdir() if x.suffix in ['.json','.csv']]
            entries += [(x,'heldout_maps/'+model+'/'+str(x.relative_to(a.run/model/'predictions/test'))) for x in files(a.run/model/'predictions/test')]
        records.append(build(a.output/(prefix+'_results.zip'),entries))
        models=[]
        for model in ['unet','deeplab','segformer']:
            for x in (a.run/model).iterdir():
                if x.is_file() and x.suffix in ['.pt','.npz','.json','.csv']:models.append((x,model+'/'+x.name))
        models += [(x,'source/root_background_v5/'+x.name) for x in files(src)]
        parent=src.parent
        for name in ['train_uploaded_upper.py','train_segformer_progress.py','prepare_deeplab_runtime.py']:
            models.append((parent/name,'source/'+name))
        models += [(x,'vendor/'+str(x.relative_to(vendor))) for x in files(vendor)]
        dockerfile=repo/'results/B3_uploaded_deeplab_v1/Dockerfile'
        if not dockerfile.exists():dockerfile=repo/'runtime/Dockerfile'
        models.append((dockerfile,'runtime/Dockerfile'))
        models.append((a.run/'selection.json','selection.json'));models.append((a.run/'stage_gate.json','stage_gate.json'))
        models.append((a.run/'unet_optimization.json','unet_optimization.json'))
        records.append(build(a.output/(prefix+'_models.zip'),models))
        initial=[]
        for model,path in [('unet','results/B3_T1T2_upper_training_v1/training/best.pt'),('segformer','results/B3_three_model_progress_v2/segformer/training/best.pt'),('deeplab','results/B3_uploaded_deeplab_v1/training/best_weights_fp16.npz')]:
            source=repo/path
            if not source.exists():source=repo/'initialization'/model/Path(path).name
            initial.append((source,'initialization/'+model+'/'+Path(path).name))
        names=repo/'results/B3_uploaded_deeplab_v1/training/weight_names.json'
        if not names.exists():names=repo/'initialization/deeplab/weight_names.json'
        initial.append((names,'initialization/deeplab/weight_names.json'))
        for name in ['best.pt','training_run.json','config.json','history.csv','validation_metrics.json']:
            source=a.run/'unet_stage1_initial'/name
            if not source.exists():source=repo/'initialization/unet_stage1_initial'/name
            initial.append((source,'initialization/unet_stage1_initial/'+name))
        records.append(build(a.output/(prefix+'_initialization.zip'),initial))
        (a.output/'README.md').write_text((a.run/'结果说明.md').read_text()+'\n\n下载文件：\n\n'+
            '\n'.join('- ['+r['filename']+']('+r['filename']+') — '+str(round(r['bytes']/1024**2,1))+' MiB' for r in records)+'\n',encoding='utf-8')
    c.js(a.output/'artifact_manifest.json',records)
    print(json.dumps(records),flush=True)
if __name__=='__main__':main()
