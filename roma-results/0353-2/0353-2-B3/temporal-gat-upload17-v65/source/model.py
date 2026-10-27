import torch
from torch import nn
import torch.nn.functional as F
class Attention(nn.Module):
 def __init__(self,dim=96,heads=4,edge_dim=8):
  super().__init__();self.heads=heads;self.d=dim//heads;self.w=nn.Linear(dim,dim,bias=False);self.a_src=nn.Parameter(torch.empty(heads,self.d));self.a_dst=nn.Parameter(torch.empty(heads,self.d));self.edge=nn.Linear(edge_dim,heads,bias=False);self.out=nn.Linear(dim,dim);self.norm=nn.LayerNorm(dim);nn.init.xavier_uniform_(self.a_src);nn.init.xavier_uniform_(self.a_dst)
 def forward(self,x,edge_index,edge_attr):
  src,dst=edge_index;h=self.w(x).reshape(len(x),self.heads,self.d);e=F.leaky_relu((h[src]*self.a_src).sum(-1)+(h[dst]*self.a_dst).sum(-1)+self.edge(edge_attr),.2)
  index=dst[:,None].expand(-1,self.heads);mx=torch.full((len(x),self.heads),-torch.inf,device=x.device);mx.scatter_reduce_(0,index,e.detach(),reduce='amax',include_self=True);a=(e-mx[dst]).exp();den=torch.zeros_like(mx).scatter_add_(0,index,a);a=a/den[dst].clamp_min(1e-8);a=F.dropout(a,.1,self.training);agg=torch.zeros_like(h);agg.index_add_(0,dst,a[:,:,None]*h[src]);return self.norm(x+F.dropout(F.elu(self.out(agg.reshape(len(x),-1))),.1,self.training))
class Model(nn.Module):
 def __init__(self,input_dim,kind):
  super().__init__();self.kind=kind;self.encoder=nn.Sequential(nn.Linear(input_dim,96),nn.LayerNorm(96),nn.ELU(),nn.Dropout(.1),nn.Linear(96,96),nn.ELU());self.context=nn.ModuleList([Attention(),Attention()]) if kind=='gat' else nn.ModuleList();self.project=nn.Sequential(nn.Linear(96,64),nn.ELU());self.temporal=nn.Linear(64,3);self.relation=nn.Sequential(nn.Linear(64*2+8,96),nn.ELU(),nn.Dropout(.1),nn.Linear(96,3));nn.init.zeros_(self.temporal.weight);nn.init.zeros_(self.temporal.bias)
 def forward(self,x,edge_index,edge_attr,pairs,pair_attrs):
  h=self.encoder(x)
  for layer in self.context:h=layer(h,edge_index,edge_attr)
  z=self.project(h);delta=3*torch.tanh(self.temporal(z));a,b=pairs.T;relation=self.relation(torch.cat([(z[a]-z[b]).abs(),z[a]*z[b],pair_attrs],1));return delta,relation,z
