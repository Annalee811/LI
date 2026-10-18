"""Verify every re-analysis image, native mask and score against pinned v9 ZIPs."""
import argparse,zipfile,hashlib
from pathlib import Path
import common as c

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--archives',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();checks=[];prefix='0353-2-B3_three_model_v9/'
 def check(z,arc,f):
  digest=hashlib.sha256(z.read(prefix+arc)).hexdigest();assert c.sha(f)==digest,(arc,str(f));checks.append(dict(file=str(f.relative_to(a.inputs)),original_member=prefix+arc,sha256=digest))
 for n in c.MODELS:
  with zipfile.ZipFile(a.archives/f'0353-2-B3_v9_segmentation_{n}.zip') as z:
   for region,part in [('C3','val'),('C4','test')]:
    for t in c.PERIODS:
     check(z,f'{n}/evaluation/{part}/{t}_after.png',a.inputs/'predictions'/n/region/f'{t}_mask.png');check(z,f'{n}/probabilities/'+('val' if region=='C3' else 'test/C4')+f'/{t}_probability.npy',a.inputs/'predictions'/n/region/f'{t}_probability.npy')
 with zipfile.ZipFile(a.archives/'0353-2-B3_v9_training_data.zip') as z:
  for t in c.PERIODS:
   for kind in ['RGB','mask']:check(z,f'data/dataset/upper/{t}_{kind}.png',a.inputs/'upper'/f'{t}_{kind}.png')
  check(z,'data/original_human_progress.json',a.inputs/'original_human_progress.json')
 c.js(a.output,dict(passed=True,source_commit=c.ORIGINAL_COMMIT,all57_native_files_and_annotations_bit_identical_to_original_v9_archives=True,checks=checks));print('ALL57_INPUT_FILES_MATCH_ORIGINAL_V9_ARCHIVES',flush=True)
if __name__=='__main__':main()
