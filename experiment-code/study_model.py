"""Controlled fresh-head/module study; no image-specific learnable node table."""
import torch
from torch import nn
from torch.nn import functional as F

class ConvLSTM(nn.Module):
 def __init__(self):super().__init__();self.gates=nn.Conv2d(16,32,3,padding=1)
 def forward(self,x,h,c):
  i,f,o,g=self.gates(torch.cat([x,h],1)).chunk(4,1);c=torch.sigmoid(f+1)*c+torch.sigmoid(i)*torch.tanh(g);return torch.sigmoid(o)*torch.tanh(c),c
class Layer(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Linear(64,64,bias=False);self.source=nn.Parameter(torch.empty(4,16));self.destination=nn.Parameter(torch.empty(4,16));self.edge=nn.Linear(8,4,bias=False);self.out=nn.Linear(64,64);self.norm=nn.LayerNorm(64);nn.init.xavier_uniform_(self.source);nn.init.xavier_uniform_(self.destination)
 def forward(self,h,index,attrs,gates):
  a,b=index;v=self.project(h).reshape(len(h),4,16);s=F.leaky_relu((v[a]*self.source).sum(-1)+(v[b]*self.destination).sum(-1)+self.edge(attrs),.2)+gates.clamp_min(.03).log()[:,None];idx=b[:,None].expand(-1,4);mx=torch.full((len(h),4),-torch.inf,device=h.device);mx.scatter_reduce_(0,idx,s.detach(),reduce='amax',include_self=True);w=(s-mx[b]).exp();den=torch.zeros_like(mx).scatter_add_(0,idx,w);w=w/den[b].clamp_min(1e-8);agg=torch.zeros_like(v);agg.index_add_(0,b,F.dropout(w,.05,self.training)[:,:,None]*v[a]);return self.norm(h+F.elu(self.out(agg.reshape(len(h),-1))))
class PairHead(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(114,64),nn.LayerNorm(64),nn.ELU(),nn.Linear(64,32),nn.ELU(),nn.Linear(32,1))
 def forward(self,x):return self.net(x)[:,0]
class Network(nn.Module):
 def __init__(self,variant):
  super().__init__();self.variant=variant;self.frame_encoder=nn.Sequential(nn.Conv2d(8,8,3,padding=1),nn.GroupNorm(4,8),nn.ELU());self.memory=ConvLSTM()if variant not in ['no_lstm','current_only']else None;self.encoder=nn.Sequential(nn.Linear(40,64),nn.LayerNorm(64),nn.ELU(),nn.Linear(64,64),nn.ELU());self.layers=nn.ModuleList([Layer(),Layer()])if variant not in ['no_gat','current_only']else nn.ModuleList();self.identity=nn.Sequential(nn.Linear(64,48),nn.ELU());self.connection=nn.Sequential(nn.Linear(104,64),nn.ELU(),nn.Linear(64,1));self.pixel=nn.Sequential(nn.Linear(79,48),nn.ELU(),nn.Linear(48,3));self.root_state=nn.Linear(64,1);nn.init.zeros_(self.pixel[-1].weight);nn.init.zeros_(self.pixel[-1].bias);nn.init.zeros_(self.root_state.weight);nn.init.zeros_(self.root_state.bias)
 def forward(self,d,external=False):
  x=d['x'];extra=d['extra']
  if self.variant=='current_only':
   x=x.clone();x[:,7]=0;x[:,14:]=0;extra=extra.clone();extra[:,3]=0
  frames=d['frames'];T=frames.shape[0];masks=d['partition_frame_masks'];B=len(masks);allframes=(frames[:,None]*masks[None,:,None]).reshape(T*B,8,*frames.shape[-2:]);amp=frames.device.type=='cpu' and torch.cpu._is_avx512_bf16_supported()
  with torch.autocast('cpu',dtype=torch.bfloat16,enabled=amp):
   latent=self.frame_encoder(allframes).reshape(T,B,8,*frames.shape[-2:]);h=torch.zeros_like(latent[0]);c=h.clone();maps=[]
   for t in range(T):
    if self.memory is None:h=latent[t]
    else:h,c=self.memory(latent[t],h,c)
    maps.append(h.float())
  context=x.new_zeros((len(x),8));phase=d['phase'];part=d['part'];centers=d['centers'];width=d.get('native_width',2541);height=d.get('native_height',1108)
  for t in range(T):
   for b in range(B):
    ids=torch.nonzero((phase==t)&(part==b)).flatten()
    if not len(ids):continue
    grid=torch.stack([centers[ids,0]/(width-1)*2-1,centers[ids,1]/(height-1)*2-1],1)[None,None];v=F.grid_sample(maps[t][b:b+1],grid,align_corners=True)[0,:,0].T;context=context.index_copy(0,ids,v)
  h=self.encoder(torch.cat([x,context,extra],1));pairs=d['pairs'];attrs=d['attrs'];a,b=pairs.T;same=phase[a]==phase[b];n=torch.arange(len(x));index=torch.cat([pairs.T,pairs[same].flip(1).T,torch.stack([n,n])],1);rev=attrs[same].clone();rev[:,1:3]*=-1;ea=torch.cat([attrs,rev,attrs.new_zeros((len(x),8))]);prior=.1+.9*torch.exp(-attrs[:,0].square()/2)*(.25+.75*attrs[:,3].square());prior=torch.where(attrs[:,5]>.5,torch.ones_like(prior),prior);gate=torch.cat([prior,prior[same],prior.new_ones(len(x))]);sym=attrs.clone();sym[:,1:3]=sym[:,1:3].abs()
  for layer in self.layers:
   ident=self.identity(h)
   if self.variant=='no_gates':affinity=gate.new_ones(len(a))
   else:
    affinity=self.connection(torch.cat([(ident[a]-ident[b]).abs(),ident[a]*ident[b],sym],1))[:,0].sigmoid();affinity=torch.where(attrs[:,5]>.5,torch.ones_like(affinity),.05+.95*affinity)
   h=layer(h,index,ea,gate*torch.cat([affinity,affinity[same],affinity.new_ones(len(x))]))
  ident=self.identity(h);links=self.connection(torch.cat([(ident[a]-ident[b]).abs(),ident[a]*ident[b],sym],1))[:,0];return h,links,F.normalize(ident,dim=1)
 def pixel_logits(self,h,node,pf):
  if self.variant=='current_only':pf=pf.clone();pf[:,:3]=-1.0986122886681098;pf[:,8:13]=0
  return pf[:,:3]+3*torch.tanh(self.pixel(torch.cat([h[node],pf],1)))
def pool(h,node,ptr):
 lengths=ptr[1:]-ptr[:-1];ids=torch.repeat_interleave(torch.arange(len(lengths)),lengths);out=h.new_zeros((len(lengths),h.shape[1]));out.index_add_(0,ids,h[node]);return out/lengths.clamp_min(1)[:,None]
def initialize(variant,seed):
 torch.manual_seed(seed);base=Network('full');shared=base.state_dict();torch.manual_seed(seed);net=Network(variant);net.load_state_dict({k:v for k,v in shared.items()if k in net.state_dict()},strict=True);torch.manual_seed(seed+31);pair=PairHead();return net,pair
