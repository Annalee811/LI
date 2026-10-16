#!/usr/bin/env python3
"""Third-model comparison on exactly the existing user-labelled grouped dataset."""
from pathlib import Path
import argparse,csv,json,shutil
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SegformerConfig,SegformerForSemanticSegmentation
import train_uploaded_upper as core

class RootSegFormer(nn.Module):
    architecture_description='SegFormer-B0: MiT [32,64,160,256], depths [2,2,2,2], heads [1,2,5,8], MLP decoder 256, random initialization'
    def __init__(self,width=None):
        super().__init__()
        config=SegformerConfig(num_channels=3,num_labels=1,hidden_sizes=[32,64,160,256],depths=[2,2,2,2],num_attention_heads=[1,2,5,8],sr_ratios=[8,4,2,1],decoder_hidden_size=256)
        self.net=SegformerForSemanticSegmentation(config)
    def forward(self,x):
        return F.interpolate(self.net(pixel_values=x).logits,size=x.shape[-2:],mode='bilinear',align_corners=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=80);a=p.parse_args()
    torch.set_num_threads(4);out=a.output;out.mkdir(parents=True,exist_ok=True);(out/'dataset').mkdir(exist_ok=True)
    for name in ['manifest.csv','dataset_summary.json']:shutil.copyfile(a.dataset/'dataset'/name,out/'dataset'/name)
    with (out/'dataset/manifest.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['column','x','y','width','height','root_pixels','ignored_pixels']:r[k]=int(r[k])
    arrays={t:(np.array(Image.open(a.dataset/f'dataset/upper/{t}_RGB.png').convert('RGB')),{k:np.array(Image.open(a.dataset/f'dataset/upper/{t}_{k}.png').convert('L')) for k in core.LAYERS}) for t in ['T1','T2']}
    args=SimpleNamespace(output=out,width=12,threads=4,batch=4,epochs=a.epochs,patience=18,lr=.001)
    model,device,threshold,run=core.train(args,rows,arrays,RootSegFormer,'SegFormer-B0')
    summary=json.loads((out/'dataset/dataset_summary.json').read_text());reports={}
    for split in ['val','test']:
        maps=core.fused_columns(model,rows,arrays,split,device);metric,per=core.evaluate_maps(maps,threshold);reports[split]={'micro_metrics':metric,'per_period_columns':per,'threshold':threshold,'n_unique_evaluated_pixels':sum(g.size for pr,g in maps.values())}
        folder=out/'predictions'/split;folder.mkdir(parents=True,exist_ok=True)
        for (t,c),(pr,g) in maps.items():
            Image.fromarray((pr>=threshold).astype('uint8')*255).save(folder/f'{t}_C{c}_prediction.png');Image.fromarray(g).save(folder/f'{t}_C{c}_human_mask.png');np.save(folder/f'{t}_C{c}_probability.npy',pr.astype(np.float16))
    core.save_json(out/'metrics.json',dict(model='SegFormer-B0',validation=reports['val'],spatial_holdout_test=reports['test'],manifest_sha256=run['manifest_sha256'],labels='identical T1/T2 upper manual masks as the two completed models',temporal_classification=False))
    print('SEGFORMER_FINAL',json.dumps(reports),flush=True)
if __name__=='__main__':main()
