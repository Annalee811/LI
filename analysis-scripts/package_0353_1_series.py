#!/usr/bin/env python3
"""Finalize shared-effective RGB, mark failed chains, and package result-only outputs."""
from pathlib import Path
import sys,json,hashlib
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evaluate_continuous_registration as v
ROOT=Path.cwd()/'results/microtube_registration';ARCHIVE=ROOT.parent
POSITIONS=['0353-1-B3','0353-1-Z2'];MAX=100*1024**2


def zip_files(path,files,base):
 with ZipFile(path,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for f in files:z.write(f,f.relative_to(base))
 with ZipFile(path) as z:assert z.testzip() is None
 assert path.stat().st_size<MAX,(path,path.stat().st_size)
 return dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest())


def shared_rgb(p,periods,reference,mask_name,summary_file):
 d=json.loads((p/summary_file).read_text())
 crop=d.get('largest_common_geometric_horizontal_rectangle_xyxy',d.get('crop_xyxy'))
 x0,y0,x1,y1=crop;mask=np.array(Image.open(p/mask_name))>0;m=mask[y0:y1,x0:x1]
 views=[]
 for i in periods:
  n=f'T{i}_registered_common_rectangle_RGB.png' if reference=='T1' else f'T{i}_common_rectangle_RGB.png'
  a=np.array(Image.open(p/n).convert('RGB'));alpha=m.astype(np.uint8)*255
  rgba=np.dstack([a,alpha]);rgba[~m,:3]=0
  f=p/f'T{i}_strict_common_effective_RGBA.png';Image.fromarray(rgba).save(f,dpi=(600,600),compress_level=5)
  back=np.full_like(a,245);back[m]=a[m];views.append(back)
  with Image.open(f) as im:
   arr=np.array(im);np.testing.assert_array_equal(arr[:,:,3],alpha);np.testing.assert_array_equal(arr[m,:3],a[m])
 oldout=v.OUT;v.OUT=p
 v.montage(views,[f'T{i} 严格共同有效RGB' for i in periods],'strict_common_effective_RGB_montage.png',f'{reference}坐标；其余区域透明/灰底，不参与定量比较')
 v.OUT=oldout
 if reference=='T1' and not m.any():
  # Standalone figures must not imply that an empty effective mask establishes alignment.
  for f in sorted(p.glob('0[1-6]_*.png')):
   im=Image.open(f).convert('RGB');canvas=Image.new('RGB',(im.width,im.height+64),'white');canvas.paste(im,(0,64));draw=ImageDraw.Draw(canvas)
   font=ImageFont.truetype(v.FONT,29)
   draw.text((18,9),'T1起始连续配准失败：共同有效像素为0；以下仅为模型估计诊断图',font=font,fill=(175,35,35))
   canvas.save(f,dpi=(600,600),compress_level=5)
  d['four_period_diagnostic_only']=True;d['warning']='T1 to T2 failed; effective four-period common region is empty. Estimated RGB warps/rectangle are diagnostic only, not reliable registration.'
  (p/summary_file).write_text(json.dumps(d,ensure_ascii=False,indent=2))
  oldout=v.OUT;v.OUT=p;v.write_report(d,[x['summary'] for x in d['cumulative_evaluations']]);v.OUT=oldout
 print('PASS: exact original-colour RGB within common masks',p.name,int(m.sum()),flush=True)
 return d


def main():
 rows=[];manifest=[]
 for pos in POSITIONS:
  p=ROOT/f'{pos}_final_continuous_evaluation'
  d=shared_rgb(p,[1,2,3,4],'T1','four_period_common_effective_mask_T1.png','continuous_evaluation.json')
  core=[f for f in p.iterdir() if f.is_file() and f.suffix in ['.png','.csv','.pdf','.json'] and f.name!='validation.json']
  extra=[]
  if pos=='0353-1-Z2':
   sup=ROOT/f'{pos}_T2_three_period_supplement'
   sd=shared_rgb(sup,[2,3,4],'T2','three_period_common_effective_mask_T2.png','three_period_summary.json')
   # Copy the requested three-period supplement into a distinct named subfolder in the Z2 package.
   import shutil
   dst=p/'T2-T4_supplement';shutil.copytree(sup,dst,dirs_exist_ok=True)
   extra=[f for f in dst.iterdir() if f.is_file()]
  else:sd=None
  arc=ARCHIVE/f'{pos}_continuous_registration_results.zip'
  # The core package includes the independently cropped RGB and exact effective-region RGB.
  manifest.append(zip_files(arc,sorted(core+extra),p))
  native=sorted((p/'native_T1_results').glob('*.png'))
  nativearc=ARCHIVE/f'{pos}_full_T1_canvas_RGB_and_masks.zip'
  manifest.append(zip_files(nativearc,native,p))
  field=p/'T1-T4_continuous_dense_fields.h5';assert field.stat().st_size<MAX
  manifest.append(dict(position=pos,file=field.name,bytes=field.stat().st_size,sha256=hashlib.file_digest(field.open('rb'),'sha256').hexdigest()))
  last=d['cumulative_evaluations'][-1]['summary']
  rows.append(dict(position=pos,reference='T1',periods='T1→T2→T3→T4',
   geometric_common_pct=d['common_geometric_pct_T1'],effective_common_pixels=d['common_effective_area_T1_px2'],effective_common_pct=d['common_effective_pct_T1'],
   root_candidate_effective_common_pct=d.get('root_candidate_common_effective_pct'),crop_width=d['rectangle_size_wh'][0],crop_height=d['rectangle_size_wh'][1],
   effective_pct_inside_crop=100*d['rectangle_effective_fraction'],automatic_endpoint_mean_T1_px=last['automatic_mean_error_T1_px'],
   automatic_qc_pass=d['full_chain_automatic_qc_pass'],supplement_T2_T4_effective_pct=sd['common_effective_pct_T2'] if sd else None))
 series=ROOT/'0353-1_series_results';series.mkdir(exist_ok=True)
 v.save_csv(series/'0353-1_series_summary.csv',rows)
 (series/'0353-1_series_summary.json').write_text(json.dumps(dict(series='0353-1',positions=rows,root_priority='Conservative curvilinear root-support proxies prioritized in coarse alignment (root:soil=3:1 when joint fitting), soil fallback; not verified semantic root segmentation.',output_colour='Original RGB preserved; any contrast normalization is inference only; outputs sampled once from original images.',manual_landmark_TRE=None,packages=manifest),ensure_ascii=False,indent=2))
 # Side-by-side sequence previews, including the usable supplement as a separately labelled third row.
 previews=[(POSITIONS[0],ROOT/f'{POSITIONS[0]}_final_continuous_evaluation/01_registered_sequence_composite.png'),
           (POSITIONS[1]+'（四期失败，仅诊断）',ROOT/f'{POSITIONS[1]}_final_continuous_evaluation/01_registered_sequence_composite.png'),
           (POSITIONS[1]+' T2→T3→T4 补充',ROOT/f'{POSITIONS[1]}_T2_three_period_supplement/01_registered_sequence_composite.png')]
 canvas=Image.new('RGB',(2400,1100),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype(v.FONT,34)
 for j,(label,path) in enumerate(previews):
  im=Image.open(path).convert('RGB');im.thumbnail((2380,290));y=j*365
  draw.text((20,y+5),label,font=font,fill=(28,35,42));canvas.paste(im,(10,y+65))
 canvas.save(series/'0353-1_series_preview.png',dpi=(300,300),compress_level=5)
 print(json.dumps(dict(summary=rows,packages=manifest),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
