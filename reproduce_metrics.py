"""Replay two-class confusion counts from the supplied fixed binary GT/predictions."""
from pathlib import Path
import argparse,json
import numpy as np
from PIL import Image
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists()and any(a.output.iterdir()):raise ValueError('Choose a new empty output directory.')
 a.output.mkdir(parents=True,exist_ok=True);root=a.bundle;truth=np.load(root/'reproduction/heldout_truth.npz');expected=json.loads((root/'metrics/All_two_class_confusions.json').read_text());checks=[]
 for group,folder in [('internal','internal'),('historical','historical'),('paired_decoder','decoder')]:
  for job,ref in expected[group].items():
   cm=np.zeros((2,2),np.int64)
   for t in ['T2','T3','T4']:
    pred=np.array(Image.open(root/'reproduction'/folder/job/(t+'_two_classes.png'))).ravel()[truth[t+'_pixel']];assert np.isin(pred,[1,2]).all()
    p=(pred==2).astype(np.int64);y=truth[t+'_truth'].astype(np.int64);cm+=np.bincount(y*2+p,minlength=4).reshape(2,2)
   assert np.array_equal(cm,np.array(ref['confusion'])),(group,job)
   checks.append(dict(group=group,job=job,confusion=cm.tolist(),matches=True))
 (a.output/'Reproduction_checks.json').write_text(json.dumps(dict(passed=True,evaluations=len(checks),checks=checks),indent=2));print('REPRODUCED_TWO_CLASS_EVALUATIONS',len(checks))
if __name__=='__main__':main()
