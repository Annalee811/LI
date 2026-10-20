"""Publish separate CRC-tested ordinary ZIPs, with immutable source hashes."""
import argparse,json,zipfile
from pathlib import Path
import common as c
p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--only',nargs='*');a=p.parse_args();repo=a.repository.resolve();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);prefix='0353-2-B3_temporal_v14/';entries=[];base=repo/'analysis-scripts/0353-2-B3_segmentation_pilot'
def tree(folder,accept=lambda p:True):return [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and accept(p)]
def pack(name,files,extra=None):
 dest=out/('0353-2-B3_v14_'+name+'.zip');items={str(p.relative_to(repo)):p for p in files};items.update(extra or {})
 items.setdefault(str((base/'temporal_reference_v14/README.md').relative_to(repo)),base/'temporal_reference_v14/README.md')
 if a.only is None or name in a.only:
  with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for arc,f in sorted(items.items()):z.write(f,prefix+arc)
 else:assert dest.exists(),('Missing skipped archive',name)
 assert dest.stat().st_size<99*1024**2,(name,dest.stat().st_size)
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 entries.append(dict(file=dest.name,bytes=dest.stat().st_size,SHA256=c.sha(dest),files=len(items)));print('PACKAGED',dest.name,round(dest.stat().st_size/1024**2,2),'MiB',flush=True)
pack('annotation_review',tree(run/'annotation_review')+[run/'editor_verification.json'])
files=tree(run/'data')+tree(run/'source_RGB')+[run/'verification.json']
# Reference-generation evidence; do not duplicate the large teacher RGB overlays.
files+=tree(run/'teacher_unet',lambda p:p.name.endswith(('_temporal_classes.png','_class_confidence_uint16.png')) or p.name=='inference.json')
pack('training_data',files)
for m in c.MODELS:
 files=tree(run/m,lambda p:p.suffix!='.pt')+tree(repo/'results/B3_annotation_refinement_v1'/m/'upper',lambda p:p.name.endswith(('_probability.npy','_root_mask.png')))+tree(run/'tables')+[run/'verification.json',run/'all_metrics.json',run/'editor_verification.json']
 pack('segmentation_'+m,files)
files=[]
for m in ['unet','segformer']:
 root=repo/'results/B3_annotation_refinement_v1'/m;files += [root/n for n in ['best.pt','config.json','selection.json','training_run.json']]
# All warm-start heads are tiny and are required to repeat the identical sequence.
for m in c.MODELS:files += [repo/'results/B3_temporal_v13'/m/'evaluation_head.pt'];files+=tree(run/m,lambda p:p.suffix=='.pt')
files += [repo/'results/B3_temporal_v13/unet/final_head.pt'];pack('weights_torch_and_temporal',files)
root=repo/'results/B3_annotation_refinement_v1/deeplab';pack('weights_deeplab',[root/n for n in ['best_weights_fp16.npz','weight_names.json','deeplab_model_adapted.py','config.json','selection.json','training_run.json']])
files=[]
for folder in ['temporal_reference_v14','temporal_v13','annotation_refinement_v1']:files+=tree(base/folder,lambda p:p.suffix in ['.py','.md','.txt','.sh'] or p.name=='.gitignore')
files += [repo/'analysis-scripts/dataset_identity.py',base/'prepare_deeplab_runtime.py'];files+=tree(repo/'dataset/current',lambda p:p.suffix in ['.json','.csv','.md'])
files += tree(repo/'results/B3_uploaded_deeplab_v1/vendor',lambda p:p.suffix in ['.py','.md','.txt'] or p.name=='LICENSE')
files += [repo/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/custom-images-v1/annotate.html']
extra={'runtime/Dockerfile':repo/'results/B3_v9_manuscript/original_v9_source/runtime/Dockerfile','runtime/prepare_deeplab_runtime.py':base/'prepare_deeplab_runtime.py'}
pack('complete_scripts',files,extra)
figs=tree(run/'paper_figures',lambda p:p.suffix!='.pdf' or p.stat().st_size<1024**2)
rgbfigs=[p for p in figs if p.name.startswith(('Fig1_','Fig2_'))]
pack('paper_figures_RGB',rgbfigs)
pack('paper_figures_other',[p for p in figs if p not in rgbfigs]+tree(run/'tables')+[run/'verification.json',base/'temporal_reference_v14/Plant_Phenomics_notes.md'])
# Small public audit artifacts can be inspected without downloading model ZIPs.
for f in [run/'verification.json',run/'editor_verification.json',run/'data/dataset_summary.json',run/'data/revision_provenance.json',run/'all_metrics.json']:
 __import__('shutil').copy2(f,out/f.name)
for f in tree(run/'tables'):__import__('shutil').copy2(f,out/f.name)
for n in ['README.md','Plant_Phenomics_notes.md']:__import__('shutil').copy2(base/'temporal_reference_v14'/n,out/n)
c.js(out/'downloads.json',dict(version='temporal_reference_v14',packages=entries,independent_external_test_images=0,reference='U-Net-derived; revised metrics are teacher agreement',all_archives_CRC_verified=True))
(out/'SHA256SUMS.txt').write_text(''.join(z['SHA256']+'  '+z['file']+'\n' for z in entries))
print('ALL_PACKAGES_READY',flush=True)
