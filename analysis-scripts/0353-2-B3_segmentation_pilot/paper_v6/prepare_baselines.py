#!/usr/bin/env python3
"""Preserve raw v5 validation maps and test native384 flip ensembling on C3 only."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
import torch
import protocol as p
from train import inference
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'root_background_v5'))
import common as oldprotocol
import train_torch as oldtorch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_uploaded_upper import SmallUNet

def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=Path,required=True);a.add_argument('--legacy',type=Path);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 legacy=args.legacy or args.data
 args.output.mkdir(parents=True,exist_ok=True);registry={}
 torch.set_num_threads(1);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 for name in ['unet','deeplab','segformer']:
  folder=args.output/'candidates'/name/'val';folder.mkdir(parents=True,exist_ok=True)
  original=legacy/name/'predictions/val/C3'
  if all((original/(t+'_probability.npy')).exists() for t in p.PERIODS):
   for t in p.PERIODS:shutil.copyfile(original/(t+'_probability.npy'),folder/(t+'_probability.npy'))
  else:
   assert name!='deeplab','Frozen TF baseline C3 maps must be supplied'
   model=oldtorch.create(name).to(device);ck=torch.load(legacy/name/'best.pt',map_location=device,weights_only=True);model.load_state_dict(ck['state_dict'])
   maps=oldprotocol.Data(args.data).column(inference(model,device),3);p.save_maps(folder,maps)
  run=json.loads((legacy/name/'training_run.json').read_text());registry[name]=dict(kind='legacy',model=name,checkpoint_sha256=run['checkpoint_sha256'],source_run=str(legacy))
 data=p.Data(args.data)
 ck=torch.load(legacy/'unet/best.pt',map_location=device,weights_only=True);model=SmallUNet(12).to(device);model.load_state_dict(ck['state_dict'])
 for tta in [False,True]:
  name='unet_context384'+('_tta' if tta else '');maps=data.column(inference(model,device,tta),3)
  p.save_maps(args.output/'candidates'/name/'val',maps)
  registry[name]=dict(kind='legacy_context',model='unet',tta=tta,checkpoint_sha256=p.sha(legacy/'unet/best.pt'),context=384)
 # One prespecified simple ensemble, no fitted weights.
 name='unet_segformer_mean';maps={t:(np.load(args.output/'candidates/unet/val'/(t+'_probability.npy'))+np.load(args.output/'candidates/segformer/val'/(t+'_probability.npy')))*.5 for t in p.PERIODS}
 p.save_maps(args.output/'candidates'/name/'val',maps);registry[name]=dict(kind='mean',components=['unet','segformer'],weights=[.5,.5])
 p.js(args.output/'baseline_registry.json',registry);print('BASELINES_READY',list(registry),flush=True)
if __name__=='__main__':main()
