import json,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import common as c
from prepare import OUT

def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);p=json.loads((OUT/'preparation.json').read_text());data=c.Data(OUT,p['roots'],'unet');ck=torch.load(OUT/'unet/best_temporal_head.pt',weights_only=True,map_location='cpu');model=c.AxisHead().to(memory_format=torch.channels_last);model.load_state_dict(ck['state_dict']);model.eval();full=OUT/'unet/full';full.mkdir(exist_ok=True)
 for t in c.PERIODS:
  dest=full/f'{t}_classes.png'
  if not dest.exists():
   if t=='T1':pred=np.where(data.gate[t],4,0).astype(np.uint8)
   else:
    q=c.predict_probability(model,data.features[t],torch.device('cpu'));assert np.allclose(q.sum(0),1,atol=1e-5);tmp=full/(t+'_probability.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,probability=q.astype(np.float16))
    os.replace(tmp,full/(t+'_probability.npz'));pred=np.where(data.gate[t],q.argmax(0)+1,0).astype(np.uint8)
   Image.fromarray(pred).save(dest)
  assert np.array_equal(np.array(Image.open(dest))>0,data.gate[t]);print('INFERRED',t,'geometry unchanged',flush=True)
 c.js(OUT/'full_inference_verification.json',dict(passed=True,reference='24',selected_epoch=ck['epoch'],binary_unchanged=True))
if __name__=='__main__':main()
