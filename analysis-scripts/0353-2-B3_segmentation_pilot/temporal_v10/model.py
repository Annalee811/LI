"""Shared native-resolution temporal head for each of the three frozen root models."""
import torch
from torch import nn
from torch.nn import functional as F

class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(4,b),nn.SiLU())
class TemporalHead(nn.Module):
    def __init__(self):
        super().__init__();self.a=Block(11,8);self.b=Block(8,16);self.c=Block(16,32);self.middle=Block(32,48)
        self.u2=Block(80,32);self.u1=Block(48,16);self.u0=Block(24,8);self.out=nn.Conv2d(8,3,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));m=self.middle(F.avg_pool2d(c,2))
        def up(v,s):return torch.cat([F.interpolate(v,size=s.shape[-2:],mode='bilinear',align_corners=False),s],1)
        return self.out(self.u0(up(self.u1(up(self.u2(up(m,c)),b)),a)))

def loss(logits,gt,weights):
    valid=(gt>0)&(gt<255)
    if not valid.any():return logits.sum()*0
    labels=gt.long()-1;labels[~valid]=255
    ce=F.cross_entropy(logits,labels,weight=weights,ignore_index=255)
    prob=logits.softmax(1);d=[]
    for k in [1,2,3]:
        g=(gt==k).float();p=prob[:,k-1]*valid
        if g.sum()>0:d.append(1-(2*(p*g).sum()+1)/(p.sum()+g.sum()+1))
    return ce+.5*torch.stack(d).mean()
