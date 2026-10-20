"""Export native RGB, overlays, categorical roots and transparent cutouts of reference."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from predict import products
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
for t in ['T1','T2','T3','T4']:
 rgb=np.array(Image.open(a.run/'data/upper'/f'{t}_RGB.png'));gt=np.array(Image.open(a.run/'data/revised'/f'{t}_classes.png'));products(a.run/'annotation_review/reference_visuals',t,rgb,gt>0,gt)
