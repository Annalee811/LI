#!/usr/bin/env python3
"""Publish two independently usable ZIPs: data/results/U-Net, and uploaded TF model."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile

p=argparse.ArgumentParser();p.add_argument('--unet',type=Path,required=True);p.add_argument('--deeplab',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=Path(__file__).resolve().parent;a.output.mkdir(parents=True,exist_ok=True)
assert json.loads((a.unet/'verification.json').read_text())['failed']==0
assert json.loads((a.unet/'two_model_verification.json').read_text())['failed']==0
assert json.loads((a.unet/'model_comparison.json').read_text())['same_test_masks']
assert (a.deeplab/'training/training_run.json').exists()
first=a.unet;second=a.deeplab
for folder in [first,second]:(folder/'source').mkdir(exist_ok=True)
for name in ['train_uploaded_upper.py','verify_uploaded_upper.py','train_uploaded_deeplab_tf1.py','infer_uploaded_deeplab_tf1.py','compare_uploaded_upper_models.py','package_upper_two_models.py','prepare_deeplab_runtime.py']:
    shutil.copyfile(source/name,first/'source'/name)
for name in ['train_uploaded_deeplab_tf1.py','infer_uploaded_deeplab_tf1.py','prepare_deeplab_runtime.py']:
    shutil.copyfile(source/name,second/'source'/name)
# Legacy attempts/benchmarks are deliberately not part of the result artifact.
first_files=[p for p in first.rglob('*') if p.is_file() and p.name!='SHA256.json' and '__pycache__' not in p.parts]
second_files=[]
for prefix in ['vendor','source']:
    second_files.extend(p for p in (second/prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
for name in ['README.md','Dockerfile','source_provenance.json']:second_files.append(second/name)
for name in ['best_weights_fp16.npz','weight_names.json','config.json','history.csv','best_epoch.json','validation_thresholds.csv','training_run.json','metrics.json','per_column_metrics.csv','deeplab_model_adapted.py']:
    second_files.append(second/'training'/name)
records=[]
for folder,files,rootname,zipname in [(first,first_files,'0353-2-B3_upper_T1T2_results','0353-2-B3_upper_T1T2_data_and_results.zip'),(second,second_files,'0353-2-B3_DeepLabV3plus_TF112','0353-2-B3_DeepLabV3plus_TF112_model.zip')]:
    hashes={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    (folder/'SHA256.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2)+'\n')
    archive=a.output/zipname
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in [*files,folder/'SHA256.json']:z.write(f,Path(rootname)/f.relative_to(folder))
    assert archive.stat().st_size<100*1024**2,'GitHub regular file cap'
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    records.append(dict(file=zipname,bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),entries=len(files)+1))
for name in ['model_comparison.csv','model_comparison.json','结果说明.txt','two_model_verification.json','verification.json','README.md']:
    shutil.copyfile(first/name,a.output/name)
for name in ['dataset_summary.json']:
    shutil.copyfile(first/'dataset'/name,a.output/name)
for name in ['train_uploaded_upper.py','verify_uploaded_upper.py','train_uploaded_deeplab_tf1.py','infer_uploaded_deeplab_tf1.py','compare_uploaded_upper_models.py','package_upper_two_models.py','prepare_deeplab_runtime.py']:
    shutil.copyfile(source/name,a.output/name)
(a.output/'artifact_sha256.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records))
