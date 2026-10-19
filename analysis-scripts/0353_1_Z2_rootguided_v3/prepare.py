"""Prepare user-guided observation bands, preserving native original RGB."""
import argparse,json,hashlib,os
from pathlib import Path
from PIL import Image

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True);pos='0353-1-Z2';out=a.work/'search/inputs/raw'/pos;out.mkdir(parents=True,exist_ok=True);orig=a.work/'inputs/raw'/pos;orig.mkdir(parents=True,exist_ok=True)
 f={1:[.01,.06,.995,.75],2:[.01,.015,.995,.56],3:[.005,.43,.995,.995],4:[.005,.53,.995,.995]};meta={}
 for t,b in f.items():
  src=next(a.raw.glob(f'T{t}.*')).resolve();im=Image.open(src).convert('RGB');box=[round(v*im.size[i%2]) for i,v in enumerate(b)];dst=orig/src.name
  if dst.is_symlink():assert dst.resolve()==src
  elif not dst.exists():dst.symlink_to(src)
  else:assert sha(dst)==sha(src)
  crop=im.crop(box);p=out/f'T{t}.png';crop.save(p);meta[f'T{t}']=dict(original_file=src.name,original_sha256=sha(src),original_wh=list(im.size),fraction_box=b,native_box=box,search_crop_sha256=sha(p),scope='Broad root-layer search prior inferred from user boxed reference images; not point correspondence truth.')
 (a.work/'user_region_priors.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2),flush=True)
if __name__=='__main__':main()
