#!/usr/bin/env python3
"""Label-free inference on arbitrary RGB files; masks retain native dimensions."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from refine_three_v9 import RootUNet,create,inference,full_maps
from continuity_v8 import proposals
from evaluate_three_v9 import result

def main():
 a=argparse.ArgumentParser();a.add_argument('--model',choices=['unet','segformer','deeplab'],required=True);a.add_argument('--bundle',type=Path,required=True);a.add_argument('--rgb-dir',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--probability-dir',type=Path);args=a.parse_args();torch.set_num_threads(3)
 weights=args.bundle/args.model;sel=json.loads((weights/'postprocessing_selection.json').read_text());classifier=json.loads((weights/'link_classifier.json').read_text());device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');args.output.mkdir(parents=True,exist_ok=True)
 model=None
 if args.model!='deeplab':
  model=(RootUNet() if args.model=='unet' else create('segformer')).to(device);model.load_state_dict(torch.load(weights/'best.pt',map_location=device,weights_only=True)['state_dict']);infer=inference(model,device,args.model=='unet')
 files=sorted(f for f in args.rgb_dir.iterdir() if f.suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff']);assert files,'No RGB files';assert len({f.stem for f in files})==len(files),'Duplicate stems'
 log=[]
 for f in files:
  rgb=np.array(Image.open(f).convert('RGB'));h,w=rgb.shape[:2]
  if model is None:
   assert args.probability_dir,'DeepLab: first run predict_tf_v9.py to produce original TF model probabilities';prob=np.load(args.probability_dir/(f.stem+'_probability.npy'))
  else:
   side=384 if args.model=='unet' else 256;padded=np.pad(rgb,((0,max(0,side-h)),(0,max(0,side-w)),(0,0)),mode='edge');prob=full_maps(SimpleNamespace(rgb={f.stem:padded}),infer,side)[f.stem][:h,:w]
  assert prob.shape==(h,w) and np.isfinite(prob).all();np.save(args.output/(f.stem+'_probability.npy'),prob);_,cc=proposals(prob,rgb,sel['threshold']);mask,added,removed,links=result(prob,rgb,sel['threshold'],cc,sel['configuration'],classifier)
  for name,im in [('root_mask',mask),('added',added),('removed',removed)]:Image.fromarray(im.astype(np.uint8)*255).save(args.output/(f.stem+'_'+name+'.png'))
  rgba=np.concatenate([rgb,mask[:,:,None].astype(np.uint8)*255],axis=2);rgba[~mask,:3]=0;Image.fromarray(rgba).save(args.output/(f.stem+'_roots_RGBA.png'))
  log.append(dict(image=f.name,width=w,height=h,root_pixels=int(mask.sum()),bridges=len(links),CUDA=torch.cuda.is_available(),human_labels_read=False));print('PREDICTED',f.name,flush=True)
 (args.output/'inference.json').write_text(json.dumps(log,indent=2)+'\n')
if __name__=='__main__':main()
