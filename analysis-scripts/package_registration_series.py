#!/usr/bin/env python3
"""Package all six continuous composites, cropped original RGB, shared masks, and numeric fields."""
from pathlib import Path
import sys,argparse,json,hashlib,shutil
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np,h5py,cv2
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_continuous_registration as v
from package_0353_1_series import shared_rgb
ROOT=Path.cwd()/'results/microtube_registration';MAX=100*1024**2


def pack(path,files,base):
 with ZipFile(path,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for f in files:z.write(f,f.relative_to(base))
 with ZipFile(path) as z:assert z.testzip() is None
 if path.stat().st_size>=MAX:raise ValueError(f'Archive exceeds GitHub single-file limit: {path}')
 return dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())


def verify_native(position,p):
 raw=ROOT/'input/raw/V-变化'/position;manifest=json.loads((ROOT/position/'manifest.json').read_text())
 for name,digest in manifest['input_sha256'].items():assert hashlib.file_digest(Path(name).open('rb'),'sha256').hexdigest()==digest
 summary=json.loads((p/'continuous_evaluation.json').read_text())
 with h5py.File(p/summary.get('dense_field_filename','T1-T4_continuous_dense_fields.h5'),'r') as hf:
  for t in summary.get('period_ids',[1,2,3,4])[1:]:
   m=hf[f'T{t}/T1_grid_to_period_xy_px'][...];geo=hf[f'T{t}/geometric_mask_T1'][...]
   rawfile=next(f for f in raw.iterdir() if f.stem.upper()==f'T{t}')
   a=np.array(Image.open(rawfile).convert('RGB'));expected=cv2.remap(a,m[:,:,0],m[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);expected[~geo]=0
   saved=np.array(Image.open(p/'native_T1_results'/f'T{t}_registered_on_T1_RGB.png'));np.testing.assert_array_equal(saved,expected)
 print('PASS',position,'input hashes and original-source single RGB remap',flush=True)


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--series',required=True);parser.add_argument('--exclude',nargs='*',default=[]);args=parser.parse_args()
 positions=sorted(d.name for d in (ROOT/'input/raw/V-变化').glob(args.series+'-*') if d.name not in args.exclude)
 rows=[];packages=[]
 series=ROOT/f'{args.series}_excluding_selected_series_results';series.mkdir(exist_ok=True)
 for pos in positions:
  raw=ROOT/'input/raw/V-变化'/pos
  if sum(f.stem.upper() in ['T1','T2','T3','T4'] for f in raw.iterdir())<=2:continue
  p=ROOT/f'{pos}_final_continuous_evaluation';v.configure_position(pos)
  d=shared_rgb(p,v.PERIOD_IDS,'T1',f'{v.COMMON_PREFIX}_common_effective_mask_T1.png','continuous_evaluation.json')
  verify_native(pos,p)
  adj=ROOT/f'{pos}_registration_evaluation'
  for source,target in [('registration_metrics.csv','adjacent_registration_metrics.csv'),('evaluation_summary.json','adjacent_evaluation_summary.json'),('registration_metrics.png','adjacent_registration_metrics.png'),('registration_evaluation_report.pdf','adjacent_registration_evaluation.pdf')]:
   shutil.copy2(adj/source,p/target)
  # Save an explicit map of the effective common rectangle, separate from geometric coverage.
  x0,y0,x1,y1=d['largest_common_geometric_horizontal_rectangle_xyxy']
  m=np.array(Image.open(p/f'{v.COMMON_PREFIX}_common_effective_mask_T1.png'))[y0:y1,x0:x1]
  Image.fromarray(m).save(p/'common_rectangle_effective_mask.png',dpi=(600,600))
  strict=sorted(p.glob('*strict_common_effective*'))
  core=sorted(f for f in p.iterdir() if f.is_file() and f.suffix in ['.png','.csv','.pdf','.json'] and f.name!='validation.json' and f not in strict)
  packages.append(pack(ROOT.parent/f'{pos}_continuous_registration_results.zip',core,p))
  packages.append(pack(ROOT.parent/f'{pos}_strict_common_effective_RGB.zip',strict,p))
  native=sorted((p/'native_T1_results').glob('*.png'));packages.append(pack(ROOT.parent/f'{pos}_full_T1_canvas_RGB_and_masks.zip',native,p))
  field=p/d.get('dense_field_filename','T1-T4_continuous_dense_fields.h5')
  if field.stat().st_size>=MAX:raise ValueError(f'Combined HDF5 exceeds GitHub limit: {field}')
  packages.append(dict(position=pos,file=field.name,bytes=field.stat().st_size,sha256=hashlib.file_digest(field.open('rb'),'sha256').hexdigest()))
  last=d['cumulative_evaluations'][-1]['summary']
  rows.append(dict(position=pos,reference='T1',chain=d['chain'],period_ids=d.get('period_ids',[1,2,3,4]),missing_period_ids=d.get('missing_period_ids',[]),dense_field_filename=field.name,
   common_geometric_pct_T1=d['common_geometric_pct_T1'],common_effective_pixels=d['common_effective_area_T1_px2'],common_effective_pct_T1=d['common_effective_pct_T1'],
   root_candidate_common_effective_pct=d.get('root_candidate_common_effective_pct'),crop_width=x1-x0,crop_height=y1-y0,
   effective_pct_inside_crop=100*d['rectangle_effective_fraction'],cumulative_automatic_mean_error_T1_px=last['automatic_mean_error_T1_px'],
   common_low_confidence_pct_T1=d['common_low_confidence_pct_T1'],adjacent_auto_qc_passed_pairs=d['adjacent_qc_passed_pairs'],adjacent_pairs=d['adjacent_pairs'],adjacent_auto_qc_pct=100*d['adjacent_qc_passed_pairs']/d['adjacent_pairs'],
   full_chain_auto_qc_pass=d['full_chain_automatic_qc_pass'],all_six_composites_saved=True))
 v.save_csv(series/f'{args.series}_series_summary.csv',rows)
 (series/f'{args.series}_series_summary.json').write_text(json.dumps(dict(series=args.series,excluded=args.exclude,positions=rows,packages=packages,
  output='Unenhanced original-source RGB sampled once to T1; common-geometry crops plus separately masked strict common-effective original RGB; six composites per location.',
  root_priority='Root ridge candidates prioritized in coarse alignment, root:soil weight 3:1; no verified semantic root segmentation.',manual_landmark_TRE=None),ensure_ascii=False,indent=2))
 # A combined series preview links to the full-resolution individual outputs.
 canvas=Image.new('RGB',(2600,640*len(rows)),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(v.FONT,34)
 for j,row in enumerate(rows):
  pos=row['position']
  p=ROOT/f'{pos}_final_continuous_evaluation';im=Image.open(p/'01_registered_sequence_composite.png').convert('RGB');im.thumbnail((2580,545))
  draw.text((20,j*640+8),pos+'｜'+row['chain']+' 裁剪拼接原色RGB'+('（T3缺失）' if row['missing_period_ids']==[3] else ''),fill=(24,31,38),font=font);canvas.paste(im,(10,j*640+65))
 canvas.save(series/f'{args.series}_series_cropped_RGB_preview.png',dpi=(300,300),compress_level=5)
 print(json.dumps(dict(rows=rows,packages=packages),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
