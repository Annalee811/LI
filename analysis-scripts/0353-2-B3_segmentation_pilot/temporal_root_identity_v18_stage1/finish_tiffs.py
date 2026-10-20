"""Lossless RGB TIFF export; remove only an entirely opaque alpha channel."""
import argparse
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
for f in sorted((a.run/'paper_figures').glob('*.tif')):
 with Image.open(f) as im:
  if im.mode=='RGBA':assert im.getchannel('A').getextrema()==(255,255)
  rgb=im.convert('RGB');size=rgb.size
 rgb.save(f,format='TIFF',compression='tiff_adobe_deflate',dpi=(600,600))
 with Image.open(f) as q:assert q.mode=='RGB' and q.size==size and q.n_frames==1 and all(abs(float(v)-600)<.01 for v in q.info['dpi'])
 print('RGB_TIFF_600_DPI',f.name,f.stat().st_size,flush=True)
