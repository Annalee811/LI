#!/usr/bin/env python3
"""Export native384 paired windows; no label editing or biological sample inflation."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import protocol as p

def write_provenance(output,source):
 files={}
 for path in sorted((output/'dataset/upper').glob('*.png')):
  expected=source/'dataset/upper'/path.name;assert p.sha(path)==p.sha(expected)
  files[str(path.relative_to(output))]=p.sha(path)
 files['dataset/manifest.csv']=p.sha(output/'dataset/manifest.csv')
 p.js(output/'dataset/provenance.json',dict(source_progress_sha256=p.sha(output/'original_human_progress.json'),files=files,
  source_v5_provenance_sha256=p.sha(source/'dataset/provenance.json'),pixel_arrays_unchanged=True))
 shutil.copyfile(source/'dataset/provenance.json',output/'source_v5_provenance.json')

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 args.output.mkdir(parents=True,exist_ok=True);upper=args.output/'dataset/upper';upper.mkdir(parents=True,exist_ok=True);rows=[]
 provenance=json.loads((args.data/'dataset/provenance.json').read_text())
 for path,digest in provenance['files'].items():assert p.sha(args.data/path)==digest
 shutil.copyfile(args.data/'original_human_progress.json',args.output/'original_human_progress.json')
 for t in p.PERIODS:
  layers={}
  for key in ['RGB','mask','color','temporal','viability']:
   source=args.data/'dataset/upper'/(t+'_'+key+'.png');dest=upper/source.name;shutil.copyfile(source,dest)
   assert p.sha(source)==p.sha(dest);layers[key]=np.array(Image.open(dest))
  for col,x0 in enumerate(p.COLS):
   split=['train','train','train','val','test'][col]
   for i,(dx,y) in enumerate(p.XY):
    x=x0+dx;stem=t+'_C'+str(col)+'_'+str(i).zfill(2);paths={}
    for key,array in layers.items():
     path=args.output/'dataset/patches'/split/key/(stem+'.png');path.parent.mkdir(parents=True,exist_ok=True)
     patch=array[y:y+384,x:x+384];assert patch.shape[:2]==(384,384);Image.fromarray(patch).save(path)
     assert np.array_equal(np.array(Image.open(path)),patch);paths[key+'_path']=str(path.relative_to(args.output));paths[key+'_sha256']=p.sha(path)
    rows.append(dict(image_id=stem,window_id='C'+str(col)+'_'+str(i),period=t,column=col,split=split,x=x,y=y,width=384,height=384,**paths))
 assert len(rows)==200
 for row in rows:
  peers=[r for r in rows if r['window_id']==row['window_id']]
  assert {r['period'] for r in peers}==set(p.PERIODS) and len({r['split'] for r in peers})==1
  for q in rows:
   if row['split']!=q['split']:
    dx=max(row['x']-q['x']-384,q['x']-row['x']-384,0);dy=max(row['y']-q['y']-384,q['y']-row['y']-384,0)
    assert max(dx,dy)>=64
 p.csvout(args.output/'dataset/manifest.csv',rows)
 p.js(args.output/'dataset/summary.json',dict(source_images=4,positions=1,native_upper_size=[2541,1108],patch_size=384,
  image_patches={s:sum(r['split']==s for r in rows) for s in ['train','val','test']},
  physical_windows={s:len({r['window_id'] for r in rows if r['split']==s}) for s in ['train','val','test']},
  all_four_periods_same_group=True,guard_pixels=64,RGB_resized=False,labels_edited=False,
  original_annotations_SHA256=p.sha(args.output/'original_human_progress.json'),
  independent_external_test_images=0,test='Previously viewed C4 repeated benchmark, not a fresh independent test',
  overlap='Highly overlapping windows within each split; not independent observations. Metrics fuse probabilities and count pixels once.',
  temporal_layers='Preserved original encoding:0unclassified/T1baseline,1new,2elongation,3persistent; not trained in binary stage',
  death_class=False))
 p.js(args.output/'dataset/verification.json',dict(passed=True,lossless_png_checks=1000,period_group_and_guard_checks=True,
  original_progress_SHA256=p.sha(args.output/'original_human_progress.json'),source_provenance_SHA256=p.sha(args.data/'dataset/provenance.json')))
 write_provenance(args.output,args.data)
 print('NATIVE384_DATA_READY',len(rows),flush=True)
if __name__=='__main__':main()
