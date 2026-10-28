"""Fresh nonlearned paired RGB/binary geometry features; no label argument."""
import argparse, os, time
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    prep=__import__('json').loads((a.run/'preparation.json').read_text());root=Path(prep['roots'])
    rgb={t:np.array(Image.open(a.run/'data/upper'/f'{t}_RGB.png')) for t in c.PERIODS}
    records=[];tic=time.monotonic()
    for m in c.MODELS:
        out=a.run/'feature_cache'/m;out.mkdir(parents=True,exist_ok=True)
        gates={t:np.array(Image.open(root/m/'upper'/f'{t}_root_mask.png'))>0 for t in c.PERIODS}
        probs={t:np.load(root/m/'upper'/f'{t}_probability.npy') for t in c.PERIODS}
        for previous,t in zip(c.PERIODS,c.PERIODS[1:]):
            path=out/(t+'.npy')
            if not path.exists():
                f=c.old.features(rgb[previous],rgb[t],probs[previous],probs[t],gates[previous],gates[t])
                assert f.shape==(1108,2541,25) and np.isfinite(f).all()
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as h:np.save(h,f.astype(np.float16))
                os.replace(tmp,path)
            f=np.load(path,mmap_mode='r');assert f.shape==(1108,2541,25) and np.isfinite(f).all()
            records.append(dict(model=m,period=t,path=str(path),sha256=c.sha(path),shape=list(f.shape)))
            c.js(a.run/'feature_progress.json',dict(completed=len(records),total=9,last_model=m,last_period=t))
            print('FEATURE',m,t,round(time.monotonic()-tic,2),flush=True)
    c.js(a.run/'feature_verification.json',dict(passed=True,generated_from_scratch=True,label_inputs=False,old_temporal_inputs=False,learned_router=False,GAT=False,records=records))

if __name__=='__main__':main()
