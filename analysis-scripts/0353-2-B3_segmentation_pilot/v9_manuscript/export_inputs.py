"""Export immutable v9 masks/RGB/probabilities and verify the original archives."""
import argparse,shutil,zipfile
from pathlib import Path
import common as c

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--archives',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);records=[]
 for r in c.read(a.archives/'SHA256.json')['archives']:
  f=a.archives/r['file'];assert f.stat().st_size==r['bytes'] and c.sha(f)==r['SHA256'];records.append(dict(**r,url='https://raw.githubusercontent.com/Annalee811/LI/'+c.ORIGINAL_COMMIT+'/'+str(a.archives)+'/'+r['file']))
 z=zipfile.ZipFile(a.archives/'0353-2-B3_v9_paper_figures.zip');key=next(n for n in z.namelist() if n.endswith('/all_metrics.json'));import json
 assert json.loads(z.read(key))==c.read(a.run/'all_metrics.json')
 for t in c.PERIODS:
  for kind in ['RGB','mask']:
   f=out/'upper'/f'{t}_{kind}.png';f.parent.mkdir(exist_ok=True);shutil.copyfile(a.data/'dataset/upper'/f.name,f)
 for name in ['all_metrics.json','per_period_metrics.csv','root_length_pairs.csv','root_length_agreement.json','frozen_selection.json','verification.json','optimization_comparison.csv']:shutil.copyfile(a.run/name,out/'records'/name)
 shutil.copyfile(a.data/'original_human_progress.json',out/'original_human_progress.json');assert c.sha(out/'original_human_progress.json')=='27943e5f9d087bf718668bba0a39058dc9a4871de6c75c5ccc27febad4c06a09'
 for n in c.MODELS:
  for region,partition in [('C3','val'),('C4','test')]:
   folder=out/'predictions'/n/region;folder.mkdir(parents=True,exist_ok=True)
   for t in c.PERIODS:
    shutil.copyfile(a.run/n/'evaluation'/partition/f'{t}_after.png',folder/f'{t}_mask.png');shutil.copyfile(a.run/n/'probabilities'/('val' if region=='C3' else 'test/C4')/f'{t}_probability.npy',folder/f'{t}_probability.npy')
 shutil.copyfile(a.run/'data/dataset/manifest.csv',out/'records/training_patch_manifest.csv')
 c.js(out/'original_archives.json',dict(original_commit=c.ORIGINAL_COMMIT,archives=records,all_local_original_archives_match_published_SHA256=True))
 c.js(out/'integrity.json',dict(files={str(f.relative_to(out)):c.sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name!='integrity.json'},task='v9 binary root/background, unmodified original annotations; no temporal v13 labels or retraining'))
 print('V9_INPUTS_EXPORTED_AND_VERIFIED',flush=True)
if __name__=='__main__':main()
