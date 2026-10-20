"""Label-free four-period prediction: native RGB + frozen binary roots only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import common as c
from continuity import decode
from predict import products
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dataset_identity import normalize_prediction_io,policy

def choose(folder,t,suffix):
 for name in [t+suffix,t+'_RGB'+suffix]:
  f=folder/name
  if f.exists():return f
 raise FileNotFoundError((folder,t,suffix))

def main():
 p=argparse.ArgumentParser();p.add_argument('--head',type=Path,required=True);p.add_argument('--router',type=Path,required=True);p.add_argument('--baseline-head',type=Path,required=True);p.add_argument('--decoder',type=Path,required=True);p.add_argument('--rgb-dir',type=Path,required=True);p.add_argument('--root-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify-against',type=Path);a=p.parse_args();a.output,position=normalize_prediction_io(a.rgb_dir,a.output);assert not a.output.exists(),'Use a new prediction directory';a.output.mkdir(parents=True);torch.set_num_threads(3);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model=c.load_head(a.head,device);strength=json.loads(a.decoder.read_text())['selected_strength'];router=json.loads(a.router.read_text());selection=json.loads(a.decoder.read_text());baseline=c.load_head(a.baseline_head,device);previous=None;records=[];axesfolder=a.output/'axes';axesfolder.mkdir()
 for t in c.PERIODS:
  rgbfile=a.rgb_dir/(t+'_RGB.png');assert rgbfile.exists(),'This entry point requires four observed adjacent periods; do not invent missing dates';rgb=np.array(Image.open(rgbfile));assert rgb.ndim==3 and rgb.shape[2]==3 and rgb.dtype==np.uint8;probfile=choose(a.root_dir,t,'_probability.npy');maskfile=choose(a.root_dir,t,'_root_mask.png');rp=np.load(probfile);mask=np.array(Image.open(maskfile))>0;assert rp.shape==mask.shape==rgb.shape[:2];axis=c.axes.build(mask,rgb,previous['axis'] if previous else None,router=router)
  if previous is None:classes=mask.astype('uint8')*4;pr=None;axisrecords=[]
  else:
   assert previous['rgb'].shape==rgb.shape,'Use aligned common-coordinate RGBs, without hidden resizing'
   link=c.axes.match(previous['axis'],axis,previous['rgb'],rgb);base=c.old.features(previous['rgb'],rgb,previous['prob'],rp,previous['mask'],mask);extra=np.concatenate([link['context'],c.axes.root_contrast(rgb)[...,None],c.axes.root_contrast(previous['rgb'])[...,None]],2);features=np.concatenate([base,extra],2).astype('float16');pr=c.predict_probability(model,features,device).astype('float16').astype('float32');bp=c.predict_probability(baseline,features,device).astype('float16').astype('float32');alpha=selection['new_head_weight'];pr=alpha*pr+(1-alpha)*bp;pr*=np.array([1,selection['new_factor'],selection['growth_factor']],np.float32)[:,None,None];pr/=pr.sum(0,keepdims=True);classes,axisrecords=decode(pr,previous['axis'],axis,link,strength,selection['decoder']);del base,extra,features
  products(a.output,t,rgb,mask,classes,pr);c.axes.save(axesfolder/(t+'.npz'),axis);c.js(a.output/(t+'_root_axes.json'),axisrecords);Image.fromarray(axis['axis_map'].astype('uint16')).save(a.output/(t+'_root_axis_ids_uint16.png'));Image.fromarray(axis['uncertain'].astype('uint8')*255).save(a.output/(t+'_crossing_review_mask.png'))
  if a.verify_against is not None:assert np.array_equal(classes,np.array(Image.open(a.verify_against/(t+'_temporal_classes.png')))),('Label-free checkpoint reload differs',t)
  records.append(dict(period=t,previous=previous['period'] if previous else None,RGB_SHA256=c.sha(rgbfile),root_probability_SHA256=c.sha(probfile),root_mask_SHA256=c.sha(maskfile),axes=len(axis['paths']),mixed_new_growth_axes=sum(q['new_nodes']>0 and q['growth_nodes']>0 for q in axisrecords)))
  previous=dict(period=t,rgb=rgb,prob=rp,mask=mask,axis=axis);print('LABEL_FREE_PREDICTED',t,flush=True)
 c.js(a.output/'inference.json',dict(position_id=position,identity_policy=policy()['version'],manual_labels_read=False,router_SHA256=c.sha(a.router),baseline_head_SHA256=c.sha(a.baseline_head),new_head_weight=selection['new_head_weight'],manual_labels_used_for_decoder_selection_here=False,decoder_strength_from_fixed_prior_validation=strength,head_SHA256=c.sha(a.head),decoder_SHA256=c.sha(a.decoder),device=str(device),CUDA=torch.cuda.is_available(),RGB_resized=False,RGB_recolored=False,records=records,verification_against_existing_output=bool(a.verify_against)))
if __name__=='__main__':main()
