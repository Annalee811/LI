"""Separate downloadable ordinary ZIPs with source provenance and CRC checks."""
import argparse,json,shutil,zipfile
from pathlib import Path
import common as c
p=argparse.ArgumentParser();p.add_argument('--repository',type=Path,default=Path.cwd());p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=a.repository.resolve();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);base=repo/'analysis-scripts/0353-2-B3_segmentation_pilot';prefix='0353-2-B3_temporal_v17/';entries=[]
def tree(folder,accept=lambda p:True):return [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and accept(p)]
def pack(name,files,extra=None):
 items={str(f.relative_to(repo)):f for f in files};items.update(extra or {});items.setdefault(str((base/'temporal_manual_continuity_v17/README.md').relative_to(repo)),base/'temporal_manual_continuity_v17/README.md');dest=out/('0353-2-B3_v17_'+name+'.zip')
 with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for arc,f in sorted(items.items()):z.write(f,prefix+arc)
 assert dest.stat().st_size<99*1024**2,(name,dest.stat().st_size)
 with zipfile.ZipFile(dest) as z:assert z.testzip() is None
 entries.append(dict(file=dest.name,bytes=dest.stat().st_size,SHA256=c.sha(dest),files=len(items)));print('PACKAGED',dest.name,round(dest.stat().st_size/1024**2,2),'MiB',flush=True)
pack('annotation_review',tree(run/'annotation_review')+[run/'editor_verification.json'])
previous=repo/'results/B3_temporal_reference_v14'
prior_files=tree(previous/'data/upper')+tree(previous/'data/original')+tree(previous/'data/revised')+[previous/'data'/n for n in ['active_progress.json','original_human_progress.json','paired_manifest.csv','spatial_manifest.csv','dataset_summary.json','revision_provenance.json']]
for m in c.MODELS:prior_files+=tree(previous/m/'four_period_prediction',lambda p:p.name.endswith('_temporal_classes.png'))
pack('training_data',tree(run/'data')+prior_files+[run/'verification.json'])
for m in c.MODELS:
 files=tree(run/m,lambda p:'development_before_tip_fragment_rule' not in p.parts and p.suffix!='.pt' and not p.name.endswith(('_raw_probability.npz','_V14_probability.npz','_temporal_review_overlay.png')))
 files+=tree(repo/'results/B3_annotation_refinement_v1'/m/'upper',lambda p:p.name.endswith(('_probability.npy','_root_mask.png')))+tree(run/'axis_cache'/m)+tree(run/'routing'/m)+tree(run/'tables')+[run/'verification.json',run/'editor_verification.json',run/'label_free_reload'/m/'inference.json']
 pack('segmentation_'+m,files)
files=[]
for m in ['unet','segformer']:
 root=repo/'results/B3_annotation_refinement_v1'/m;files += [root/n for n in ['best.pt','config.json','selection.json','training_run.json']]
for m in c.MODELS:files+=[previous/m/n for n in ['best_temporal_head.pt','decoder_selection.json','training_run.json','config.json']];files+=tree(run/m,lambda p:p.suffix=='.pt')
pack('weights_torch_and_temporal',files)
root=repo/'results/B3_annotation_refinement_v1/deeplab';pack('weights_deeplab',[root/n for n in ['best_weights_fp16.npz','weight_names.json','deeplab_model_adapted.py','config.json','selection.json','training_run.json']])
files=[]
for folder in ['temporal_manual_continuity_v17','temporal_reference_v14','temporal_v13','annotation_refinement_v1']:files+=tree(base/folder,lambda p:p.suffix in ['.py','.md','.txt','.sh','.cjs'] or p.name=='.gitignore')
files+=[repo/'analysis-scripts/dataset_identity.py',base/'prepare_deeplab_runtime.py'];files+=tree(repo/'dataset/current',lambda p:p.suffix in ['.json','.csv','.md']);files+=tree(repo/'results/B3_uploaded_deeplab_v1/vendor',lambda p:p.suffix in ['.py','.md','.txt'] or p.name=='LICENSE');files+=[repo/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/custom-images-v1/annotate.html']
pack('complete_scripts',files,{'runtime/Dockerfile':repo/'results/B3_v9_manuscript/original_v9_source/runtime/Dockerfile','runtime/prepare_deeplab_runtime.py':base/'prepare_deeplab_runtime.py'})
figs=tree(run/'paper_figures',lambda p:p.suffix!='.pdf' or p.stat().st_size<1024**2)
pack('paper_figures',figs+tree(run/'tables')+[run/'verification.json',base/'temporal_manual_continuity_v17/Plant_Phenomics_notes.md'])
for f in [run/'verification.json',run/'editor_verification.json',run/'data/revision_provenance.json',run/'data/dataset_summary.json',run/'all_metrics.json']+tree(run/'tables'):shutil.copy2(f,out/f.name)
for n in ['README.md','Plant_Phenomics_notes.md']:shutil.copy2(base/'temporal_manual_continuity_v17'/n,out/n)
shutil.copy2(run/'integrated/0353-2-B3_temporal_integrated_v17.tif',out/'0353-2-B3_temporal_integrated_v17.tif')
shutil.copy2(run/'integrated/figure_protocol.json',out/'integrated_figure_protocol.json')
c.js(out/'downloads.json',dict(version='temporal_manual_continuity_v17',packages=entries,direct_figures=[dict(file='0353-2-B3_temporal_integrated_v17.tif',bytes=(out/'0353-2-B3_temporal_integrated_v17.tif').stat().st_size,SHA256=c.sha(out/'0353-2-B3_temporal_integrated_v17.tif'))],all_ZIP_CRC_verified=True,reference_temporal_classes_are_human=True,reference_morphology_is_U_Net_derived=True,V14_preserved=True,original_human_preserved_separately=True,independent_external_test_images=0))
(out/'SHA256SUMS.txt').write_text(''.join(v['SHA256']+'  '+v['file']+'\n' for v in entries)+c.sha(out/'0353-2-B3_temporal_integrated_v17.tif')+'  0353-2-B3_temporal_integrated_v17.tif\n');print('ALL_V17_PACKAGES_READY',flush=True)
