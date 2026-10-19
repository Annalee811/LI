"""Additional diagnostic crop/contrast attempt; never changes output RGB."""
import argparse,json,cv2,numpy as np
from pathlib import Path
from PIL import Image

def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();orig=a.work/'inputs/raw/0353-1-Z2';out=a.work/'tight_gray/inputs/raw/0353-1-Z2';out.mkdir(parents=True,exist_ok=True);fs={1:[.01,.06,.995,.75],2:[.01,.04,.995,.46],3:[.005,.60,.995,.995]};m={};cv2.setNumThreads(1)
 for t,b in fs.items():
  im=Image.open(next(orig.glob(f'T{t}.*'))).convert('RGB');box=[round(v*im.size[i%2]) for i,v in enumerate(b)];rgb=np.array(im.crop(box));g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);g=cv2.createCLAHE(3,(12,12)).apply(g);Image.fromarray(g).convert('RGB').save(out/f'T{t}.png');m[f'T{t}']=dict(native_box=box,original_wh=list(im.size),preprocessing='CLAHE grayscale for matching only; original colors retained in final RGB')
 (a.work/'tight_gray/prior.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
