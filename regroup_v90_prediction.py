"""Reusable downstream grouping. Root identities and foreground are preserved."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
from PIL import Image
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--classes',type=Path,required=True);ap.add_argument('--ownership',type=Path);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 if a.output.exists()and any(a.output.iterdir()):raise ValueError('Choose a new empty output directory; original results are preserved.')
 a.output.mkdir(parents=True,exist_ok=True);cl=np.array(Image.open(a.classes));assert cl.ndim==2 and set(np.unique(cl)).issubset({0,1,2,3,4,255})
 mapped=np.zeros(cl.shape,np.uint8);mapped[np.isin(cl,[1,3])]=1;mapped[cl==2]=2;mapped[np.isin(cl,[4,255])]=255
 assert np.array_equal(mapped>0,cl>0)and np.array_equal(mapped==2,cl==2)
 Image.fromarray(mapped).save(a.output/'two_classes.png');flagged=mapped.copy()
 if a.ownership:
  own=dict(np.load(a.ownership));assert all(v.shape==cl.shape for v in own.values())
  if 'uncertain'in own:flagged[own['uncertain']&(cl>0)]=255
  shutil.copy2(a.ownership,a.output/'original_ownership.npz')
 Image.fromarray(flagged).save(a.output/'two_classes_uncertainty_retained.png')
 for label,x in [('two_classes',mapped),('two_classes_uncertainty_retained',flagged)]:
  rgb=np.full((*x.shape,3),255,np.uint8);rgb[x==1]=(63,112,145);rgb[x==2]=(213,154,51);rgb[x==255]=(177,177,177);Image.fromarray(rgb).save(a.output/(label+'_on_white.png'))
 (a.output/'grouping_provenance.json').write_text(json.dumps(dict(input_SHA256=hashlib.sha256(a.classes.read_bytes()).hexdigest(),
  original_to_output={'persistent_1':'existing_1','extension_3':'existing_1','new_2':'new_2','baseline_4_or_unknown_255':'unassessed_255'},
  classes_only_grouped=True,foreground_preserved=True,new_pixels_unchanged=True,root_instances_not_merged=True,model_retrained=False),indent=2))
if __name__=='__main__':main()
