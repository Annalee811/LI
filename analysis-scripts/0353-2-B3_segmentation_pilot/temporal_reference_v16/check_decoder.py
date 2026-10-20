"""Verify cached width projection is exactly equivalent to saved v15 outputs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import common as c
from structured import decode
p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--roots',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
for m in c.MODELS:
 data=c.Data(a.previous,a.roots,m);strength=json.loads((a.previous/m/'decoder_selection.json').read_text())['selected_strength']
 for prev,t in zip(c.PERIODS,c.PERIODS[1:]):
  pr=np.load(a.previous/m/'four_period_prediction'/f'{t}_raw_probability.npz')['probability'].astype('float32');out,_=decode(pr,data.axes[prev],data.axes[t],data.link[t],strength);saved=np.array(Image.open(a.previous/m/'four_period_prediction'/f'{t}_temporal_classes.png'));assert np.array_equal(out,saved),(m,t);records.append(dict(model=m,period=t,exact=True));print('DECODER_EQUIVALENT',m,t,flush=True)
c.js(a.output,dict(PASS=True,records=records))
