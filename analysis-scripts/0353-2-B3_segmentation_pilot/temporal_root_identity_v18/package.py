"""Package the final two-stage run, teacher evidence and complete source snapshots."""
import argparse,json,shutil,zipfile
from pathlib import Path
from PIL import Image
import common as c

p=argparse.ArgumentParser()
p.add_argument('--repository',type=Path,default=Path.cwd())
p.add_argument('--run',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();repo=a.repository.resolve();run=a.run.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
base=repo/'analysis-scripts/0353-2-B3_segmentation_pilot';source=base/'temporal_root_identity_v18'
stage1=repo/'results/B3_temporal_root_identity_v18_stage1';previous=repo/'results/B3_temporal_manual_continuity_v17';v14=repo/'results/B3_temporal_reference_v14'
prefix='0353-2-B3_temporal_v18/';entries=[];direct=[]

def tree(folder,accept=lambda p:True):
    return [f for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts and accept(f)]

def pack(name,files,extra=None):
    items={str(f.relative_to(repo)):f for f in files};items.update(extra or {})
    items.setdefault(str((source/'README.md').relative_to(repo)),source/'README.md')
    dest=out/('0353-2-B3_v18_'+name+'.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for arc,f in sorted(items.items()):
            assert f.is_file(),f
            z.write(f,prefix+arc)
    assert dest.stat().st_size<99*1024**2,(name,dest.stat().st_size)
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None
    entries.append(dict(file=dest.name,bytes=dest.stat().st_size,SHA256=c.sha(dest),files=len(items)))
    print('PACKAGED',dest.name,round(dest.stat().st_size/1024**2,2),'MiB',flush=True)

assert json.loads((run/'verification.json').read_text())['status']=='PASS'
pack('annotation_review',tree(run/'annotation_review')+[run/'editor_verification.json'])

prior_files=[]
for folder in ['upper','original','revised','v14_reference']:prior_files+=tree(previous/'data'/folder)
prior_files += [previous/'data'/n for n in ['active_progress.json','original_human_progress.json','paired_manifest.csv','spatial_manifest.csv','dataset_summary.json','revision_provenance.json']]
prior_files += [previous/'routing/unet/router.json']
for m in c.MODELS:
    prior_files += [previous/m/'metrics.json']
    prior_files+=tree(previous/m/'four_period_prediction',lambda f:f.name.endswith('_temporal_classes.png') or m=='unet' and f.name.endswith('_class_probability.npz'))
teacher=tree(run/'teacher_axis_cache')+tree(run/'teacher_identity_preview',lambda f:f.name.endswith(('_temporal_classes.png','_root_axes.json','_class_probability.npz')))+[run/'teacher_evidence_provenance.json']
pack('training_data',tree(run/'data')+prior_files+teacher+[run/'verification.json'])

stagefiles=tree(stage1/'data')+tree(stage1/'teacher_axis_cache')+tree(stage1/'teacher_identity_preview',lambda f:f.name.endswith(('_temporal_classes.png','_root_axes.json','_class_probability.npz')))+tree(stage1/'routing')+[stage1/'teacher_evidence_provenance.json',stage1/'verification.json']
for m in c.MODELS:
    stagefiles += [stage1/m/n for n in ['history.csv','training_draws.csv','config.json','training_run.json','decoder_selection.json','baseline_validation_whole_axis.json']]
pack('stage1_training_data_and_history',stagefiles)

roots=repo/'results/B3_annotation_refinement_v1'
for m in c.MODELS:
    files=tree(run/m,lambda f:f.suffix!='.pt' and not f.name.endswith(('_raw_probability.npz','_warm_probability.npz')))
    files+=tree(roots/m/'upper',lambda f:f.name.endswith(('_probability.npy','_root_mask.png')))+tree(run/'axis_cache'/m)+tree(run/'routing'/m)+tree(run/'tables')
    files += [run/'verification.json',run/'editor_verification.json',run/'label_free_reload'/m/'inference.json']
    pack('segmentation_'+m,files)

files=[]
for m in ['unet','segformer']:
    files += [roots/m/n for n in ['best.pt','config.json','selection.json','training_run.json']]
for m in c.MODELS:
    for folder in [run/m,stage1/m]:
        files+=tree(folder,lambda f:f.suffix=='.pt' or f.name in ['config.json','training_run.json','decoder_selection.json','history.csv'])
    files+=tree(run/'warm_starts'/m)+tree(stage1/'warm_starts'/m)
    start=previous if m=='unet' else v14
    files += [start/m/n for n in ['best_temporal_head.pt','decoder_selection.json','training_run.json','config.json']]
pack('weights_torch_and_temporal',files)
pack('weights_deeplab',[roots/'deeplab'/n for n in ['best_weights_fp16.npz','weight_names.json','deeplab_model_adapted.py','config.json','selection.json','training_run.json']])

files=[]
for folder in ['temporal_root_identity_v18','temporal_root_identity_v18_stage1','temporal_manual_continuity_v17','temporal_reference_v14','temporal_v13','annotation_refinement_v1']:
    files+=tree(base/folder,lambda f:f.suffix in ['.py','.md','.txt','.sh','.cjs','.json'] or f.name=='.gitignore')
files += [repo/'analysis-scripts/dataset_identity.py',base/'prepare_deeplab_runtime.py']
files += tree(repo/'dataset/current',lambda f:f.suffix in ['.json','.csv','.md'])
files += tree(repo/'results/B3_uploaded_deeplab_v1/vendor',lambda f:f.suffix in ['.py','.md','.txt'] or f.name=='LICENSE')
files += [repo/'roma-results/0353-2/0353-2-B3/segmentation-pilot-annotation/custom-images-v1/annotate.html']
pack('complete_scripts',files,{'runtime/Dockerfile':repo/'results/B3_v9_manuscript/original_v9_source/runtime/Dockerfile','runtime/prepare_deeplab_runtime.py':base/'prepare_deeplab_runtime.py'})
pack('paper_figures',tree(run/'paper_figures')+tree(run/'tables')+[run/'verification.json',source/'Plant_Phenomics_notes.md',run/'annotation_review/selective_revision_evidence.tif'])

for f in [run/'verification.json',run/'editor_verification.json',run/'data/revision_provenance.json',run/'data/revision_summary.json',run/'data/dataset_summary.json',run/'all_metrics.json']+tree(run/'tables'):
    shutil.copy2(f,out/f.name)
for n in ['README.md','Plant_Phenomics_notes.md']:shutil.copy2(source/n,out/n)
for f in [run/'integrated/0353-2-B3_temporal_integrated_v18.tif',run/'paper_figures/FigS7_user_detail_before_after.tif',run/'paper_figures/FigS8_individual_root_routes.tif']:
    dest=out/f.name;shutil.copy2(f,dest)
    with Image.open(dest) as im:
        im.load();assert im.mode=='RGB' and im.n_frames==1 and all(abs(float(x)-600)<.01 for x in im.info['dpi'])
        size=list(im.size)
    direct.append(dict(file=dest.name,bytes=dest.stat().st_size,SHA256=c.sha(dest),dimensions=size,DPI=600))
shutil.copy2(run/'integrated/figure_protocol.json',out/'integrated_figure_protocol.json')
c.js(out/'downloads.json',dict(version='temporal_root_identity_v18',packages=entries,direct_figures=direct,all_ZIP_CRC_verified=True,temporal_reference_selectively_U_Net_supported=True,reference_morphology_is_U_Net_derived=True,semantic_pixels_changed=3581,original_human_preserved_separately=True,V17_preserved=True,all_three_final_emission_blends_retain_stage1_head=True,independent_external_test_images=0))
(out/'SHA256SUMS.txt').write_text(''.join(v['SHA256']+'  '+v['file']+'\n' for v in entries+direct))
print('ALL_V18_PACKAGES_READY',flush=True)
