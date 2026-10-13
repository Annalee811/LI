"""Package soil-structure localization and updated RoMa diagnostics without claiming validated continuity."""
from pathlib import Path
import sys,json,shutil,hashlib
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np,h5py
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_continuous_registration as v
from package_0353_1_series import shared_rgb
from package_registration_series import verify_native
ROOT=Path.cwd()/'results/microtube_registration';P=ROOT/'A017-1-Z2_final_continuous_evaluation';COARSE=ROOT/'A017-1-Z2_soil_structure_coarse_results';PAIR=ROOT/'A017-1-Z2/T2_to_T4'
v.configure_position('A017-1-Z2')
s=shared_rgb(P,[1,2,4],'T1','three_period_common_effective_mask_T1.png','continuous_evaluation.json');verify_native('A017-1-Z2',P)
for a,b in [('registered_rgb.png','T2_to_T4_local_registered_RGB.png'),('confidence_preview.png','T2_to_T4_local_confidence_preview.png'),('valid_overlap_mask.png','T2_to_T4_local_effective_mask.png')]:shutil.copy2(PAIR/a,P/b)
adj=ROOT/'A017-1-Z2_registration_evaluation'
for a,b in [('registration_metrics.csv','adjacent_registration_metrics.csv'),('evaluation_summary.json','adjacent_evaluation_summary.json'),('registration_metrics.png','adjacent_registration_metrics.png'),('registration_evaluation_report.pdf','adjacent_registration_evaluation.pdf')]:shutil.copy2(adj/a,P/b)
meta=json.loads((PAIR/'metrics.json').read_text())
comparison=dict(position='A017-1-Z2',period_ids=[1,2,4],missing_period_ids=[3],baseline_common_effective_pct_T1=0,updated_common_effective_pct_T1=s['common_effective_pct_T1'],baseline_T2_to_T4_effective_pct_target=0,updated_T2_to_T4_effective_pct_target=100*meta['effective_area_ratio'],full_chain_automatic_qc_pass=s['full_chain_automatic_qc_pass'],first_pair_accepted=False,manual_landmark_TRE=None,
 explanation='Soil crack descriptors provide coarse localization, but T1/T2 RoMa candidates have too few reliable matches. Full continuity remains unvalidated; coarse affine images and RoMa dense diagnostics have distinct meanings.',local_RoMa_metrics=meta)
(P/'soil_revision_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2))
(P/'RESULT_NOTES.txt').write_text('T1→T2→T4；T3缺失。\nsoil_structure_alignment：原始RGB经土壤裂缝结构粗对齐；A–C为观察窗口，非人工标志点真值。\n六类RoMa综合图：完整链仍无严格有效共同区域，仅供诊断。T2→T4局部RoMa有效面积占原始T4画布3.43%。\n土壤粗对齐场与RoMa变换场分别标注，不将粗对齐残差作为人工TRE。\n',encoding='utf-8')
source=PAIR/'dense_transform.npz';local_field=COARSE/'T2_to_T4_local_RoMa_fields.h5'
if not local_field.exists():
 with np.load(source) as z,h5py.File(local_field,'w') as f:
  f.attrs['field_type']='RoMa T2 -> T4, native canvases; ROI inference; outside ROI confidence0; not validated T1 continuity';f.attrs['metrics_json']=json.dumps(meta,ensure_ascii=False)
  for key in z.files:
   if 'cycle' in key or 'jacobian' in key:continue
   a=z[key];f.create_dataset(key,data=a,compression='gzip',compression_opts=6,shuffle=True,fletcher32=True,chunks=True) if a.ndim>=2 else f.create_dataset(key,data=a)
with np.load(source) as z,h5py.File(local_field,'r+') as f:
 for key in f:np.testing.assert_array_equal(f[key][...],z[key])
 f.attrs['source_dense_npz_sha256']=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
assert local_field.stat().st_size<100*1024**2
core=[f for f in P.iterdir() if f.is_file() and f.suffix in ['.png','.csv','.json','.pdf','.txt'] and f.name!='validation.json']
coarse=[f for f in COARSE.iterdir() if f.is_file() and f.name!=local_field.name]
archive=ROOT.parent/'A017-1-Z2_soil_shapes_revision_results.zip'
with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(core):z.write(f,f.name)
 for f in sorted(coarse):z.write(f,'soil_structure_alignment/'+f.name)
with ZipFile(archive) as z:assert z.testzip() is None
assert archive.stat().st_size<100*1024**2,archive.stat().st_size
native_archive=ROOT.parent/'A017-1-Z2_soil_revision_full_T1_RGB_and_masks.zip'
with ZipFile(native_archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted((P/'native_T1_results').glob('*.png')):z.write(f,'native_T1_results/'+f.name)
with ZipFile(native_archive) as z:assert z.testzip() is None
files=[archive,native_archive,local_field,P/'T1-T2-T4_continuous_dense_fields.h5']
manifest=dict(results=[dict(file=f.name,bytes=f.stat().st_size,sha256=hashlib.file_digest(f.open('rb'),'sha256').hexdigest()) for f in files],comparison=comparison)
(P/'soil_revision_download_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('PASS revision archives and lossless local fields',[(f.name,f.stat().st_size) for f in files],flush=True)
