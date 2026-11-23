from pathlib import Path
import csv,json,hashlib,zipfile,shutil
import numpy as np
from PIL import Image
import fitz
from openpyxl import load_workbook
W=Path(__file__).resolve().parent
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def archive(path,files):
 with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
  for src,rel in files:z.write(src,str(rel))
 with zipfile.ZipFile(path)as z:assert z.testzip()is None
 assert path.stat().st_size<95*2**20,(path.name,'Needs a smaller independent shard')
 print('NEW_ONLY_ARCHIVE',path.name,round(path.stat().st_size/2**20,2),'MiB',flush=True)
def main():
 D=W/'delivery';D.mkdir(exist_ok=True);rr=rows(W/'All_66_new_root_prediction_manifest.csv')
 for r in rr:
  src=Path(r['source_class_path']);assert sha(src)==r['source_class_SHA256'];pred=np.array(Image.open(src));n=pred==2
  mask=W/r['mask_path'];assert sha(mask)==r['new_mask_SHA256'];actual=np.array(Image.open(mask));assert np.array_equal(actual,n.astype(np.uint8)*255)
  rgb=np.array(Image.open(r['RGB_path']).convert('RGB'));ov=np.array(Image.open(W/r['overlay_path']));white=np.array(Image.open(W/r['new_root_RGB_path']))
  assert np.array_equal(ov[~n],rgb[~n]);assert np.array_equal(white[n],rgb[n]);assert np.all(white[~n]==255)
 assert len(rr)==66 and len(set(r['position_id']for r in rr))==18
 modelrows=rows(W/'Baseline_ablation_442_new_masks_index.csv')
 for r in modelrows:
  p=W/r['new_mask_path'];assert sha(p)==r['mask_SHA256'];src=Path('/workspace/work/V90_validation_20261123/independent_predictions')/r['source'];assert sha(src)==r['source_SHA256'];assert np.array_equal(np.array(Image.open(p))==255,np.array(Image.open(src))==2)
 assert len(modelrows)==442
 for name,pages in [('All_18_series_new_roots_only.pdf',18),('Independent_new_roots_only_comparison.pdf',7),('New_root_precision_recall_SCI.pdf',1),('B3_V66_V90_new_roots_only_SCI.pdf',1)]:
  doc=fitz.open(W/name);assert len(doc)==pages;doc.close()
 for stem in ['New_root_precision_recall_SCI','B3_V66_V90_new_roots_only_SCI']:
  for ext in ['jpg','png','tif']:
   with Image.open(W/(stem+'.'+ext))as im:assert all(abs(float(d)-600)<.1 for d in im.info['dpi'])
 wb=load_workbook(W/'New_root_Precision_Recall.xlsx',read_only=True);assert 'B3_full_image'in wb.sheetnames;wb.close()
 pending=json.loads((W/'checks/pending_independent_evaluation/New_root_evaluation_status.json').read_text());assert pending['Precision']is None and pending['Recall']is None and len(pending['missing_human_GT'])==19
 checks=dict(passed=True,new_only_masks_verified=66,comparison_new_only_masks_verified=442,only_class_2_selected=True,
  other_classes_not_merged=True,original_RGB_outside_new_pixels_unchanged=True,all_metrics_only_New_Precision_and_Recall=True,
  baseline_new_class_not_assessed=True,missing_dates_not_synthesized=True,independent_Precision=None,independent_Recall=None,
  pending_new_class_GT_frames=19,saved_binary_inputs_reproduce_26_evaluations=True,model_weights_and_human_annotations_not_modified=True)
 (W/'New_only_final_checks.json').write_text(json.dumps(checks,indent=2))
 # One self-contained binary-evaluation and figure bundle; large TIFFs/PDF atlases are separate links.
 exclude={'All_18_series_new_roots_only.pdf','Independent_new_roots_only_comparison.pdf'}
 files=[(p,Path('New_roots_only')/p.name)for p in W.iterdir()if p.is_file()and p.name not in exclude and p.suffix!='.tif']
 for folder in ['metrics','checks','B3_human_new_truth','New_only_reproduction']:
  files +=[(p,Path('New_roots_only')/p.relative_to(W))for p in(W/folder).rglob('*')if p.is_file()]
 archive(D/'New_root_figures_metrics_code.zip',files)
 # Native PNGs and identity maps are split by canonical tube; each ZIP stands alone.
 for tube in sorted(set(r['tube_id']for r in rr)):
  positions=set(r['position_id']for r in rr if r['tube_id']==tube);files=[]
  for pos in sorted(positions):files +=[(p,Path('New_roots_native')/pos/p.name)for p in(W/'all_series_new_only'/pos).iterdir()if p.is_file()]
  files +=[(W/'README.txt','README.txt'),(W/'All_66_new_root_prediction_manifest.csv','All_66_new_root_prediction_manifest.csv')]
  archive(D/('New_roots_native_'+tube+'.zip'),files)
 files=[(p,Path('baseline_ablation_new_masks')/p.relative_to(W/'baseline_ablation_new_masks'))for p in(W/'baseline_ablation_new_masks').rglob('*')if p.is_file()]
 files +=[(W/'Baseline_ablation_442_new_masks_index.csv','Baseline_ablation_442_new_masks_index.csv'),(W/'README.txt','README.txt')]
 archive(D/'Baseline_ablation_442_new_only_masks.zip',files)
 for stem in ['New_root_precision_recall_SCI','B3_V66_V90_new_roots_only_SCI']:
  for ext in ['jpg','pdf','tif','svg']:
   p=W/(stem+'.'+ext);shutil.copy2(p,D/p.name)
 for name in ['All_18_series_new_roots_only.pdf','Independent_new_roots_only_comparison.pdf','New_root_Precision_Recall.xlsx','New_root_results_and_captions.docx','New_root_results_CN.txt','README.txt','All_66_new_root_prediction_manifest.csv','New_only_final_checks.json']:
  shutil.copy2(W/name,D/name)
 for p in W.glob('*_SCI_preview.png'):shutil.copy2(p,D/p.name)
 manifest={p.name:dict(bytes=p.stat().st_size,SHA256=sha(p))for p in sorted(D.iterdir())if p.is_file()and p.name!='New_only_artifact_SHA256.json'}
 (D/'New_only_artifact_SHA256.json').write_text(json.dumps(manifest,indent=2));print('NEW_ONLY_DELIVERY_READY',len(manifest),'artifacts;',round(sum(x['bytes']for x in manifest.values())/2**20,2),'MiB',flush=True)
if __name__=='__main__':main()
