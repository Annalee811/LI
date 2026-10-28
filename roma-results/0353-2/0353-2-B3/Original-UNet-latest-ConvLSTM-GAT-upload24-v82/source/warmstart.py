"""Latest learned local module is separate from the original U-Net's frozen class baseline."""
import torch
from prepare import ROOT,sha
LATEST=ROOT/'results/B3_UNet_V75_latestCG_upload24_v81_20261028/best.pt'
CHECKPOINT_SHA256='ce29d142a03c5b4ee0bf0ca9ace14e033e5bada5e1b548c8fa24ea57769646bc'
def inherit_latest_module(model):
 assert sha(LATEST)==CHECKPOINT_SHA256;c=torch.load(LATEST,map_location='cpu',weights_only=False);assert c['epoch']==13;model.load_state_dict(c['state_dict'],strict=True)
 assert all(torch.equal(model.state_dict()[k],v) for k,v in c['state_dict'].items())
