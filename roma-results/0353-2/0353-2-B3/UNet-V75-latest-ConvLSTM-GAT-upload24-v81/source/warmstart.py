"""Inherit the latest learned correction module; keep V75 maps as a separate immutable anchor."""
import torch
from prepare import ROOT,sha
LATEST=ROOT/'results/B3_UNet_local_upload24_v79_20261028/best.pt'
CHECKPOINT_SHA256='47175a3ac53b2341a2461fb92227bbcdbfe3c2ab371f911fa459c2569f3f3728'
def inherit_latest_module(model):
 assert sha(LATEST)==CHECKPOINT_SHA256;c=torch.load(LATEST,map_location='cpu',weights_only=False);assert c['epoch']==92
 model.load_state_dict(c['state_dict'],strict=True)
 assert all(torch.equal(model.state_dict()[k],v) for k,v in c['state_dict'].items()),'Latest module inheritance mismatch'
