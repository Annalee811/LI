from pathlib import Path
import sys,numpy as np
from PIL import Image
import common as c
import argparse
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run=a.run;old=a.previous;r={}
for n in c.MODELS:
 r[n]={}
 for version,base in [('v12',old),('v13',run)]:
  cm=np.zeros((4,4),np.int64)
  for t in c.PERIODS[1:]:
   gt=np.array(Image.open(run/'data/targets'/f'{t}_temporal_target.png'));pred=np.array(Image.open(base/n/'four_period_final_fit'/f'{t}_temporal_classes.png'));cm+=c.confusion(gt,pred)
  r[n][version]=c.metrics(cm)
c.js(run/'all_upper_fit_metrics.json',dict(purpose='Fitted-image error analysis only; all upper labels train deployment heads, NOT validation or independent accuracy',reference='same revised labels for both versions',models=r));print({n:{v:r[n][v]['macro_Dice'] for v in r[n]} for n in r})
