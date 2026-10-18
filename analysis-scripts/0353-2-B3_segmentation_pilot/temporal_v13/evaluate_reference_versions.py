"""Separate the numerical effect of annotation revision from model changes."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import common as c

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();rows=[];reports={}
    for n in c.MODELS:
        reports[n]={}
        for version,root in [('v12',a.previous),('v13',a.run)]:
            reports[n][version]={}
            for reference,data in [('original',a.run/'data/original_reference'),('revised',a.run/'data')]:
                reports[n][version][reference]={}
                for split,col in [('validation',3),('repeated_benchmark',4)]:
                    cm=np.zeros((4,4),np.int64)
                    for t in c.PERIODS[1:]:
                        gt=np.array(Image.open(data/'targets'/f'{t}_temporal_target.png'))[:,c.COLS[col]:c.COLS[col]+448];pred=np.array(Image.open(root/n/'evaluation'/split/f'{t}_classes.png'));cm+=c.confusion(gt,pred)
                    m=c.metrics(cm);reports[n][version][reference][split]=m
                    for r in m['classes']:rows.append(dict(model=n,model_version=version,reference_version=reference,split=split,**r))
    c.js(a.run/'reference_version_metrics.json',reports);c.csvout(a.run/'reference_version_metrics.csv',rows);print('BOTH_REFERENCE_VERSIONS_EVALUATED',flush=True)
if __name__=='__main__':main()
