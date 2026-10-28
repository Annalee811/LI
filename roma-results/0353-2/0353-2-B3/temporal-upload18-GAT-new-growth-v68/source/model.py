"""Relation-gated GAT; class boundaries and root identity are distinct predictions."""
import torch
from torch import nn
import torch.nn.functional as F

class Layer(nn.Module):
    def __init__(self,dim=64,heads=4):
        super().__init__();self.heads=heads;self.d=dim//heads
        self.project=nn.Linear(dim,dim,bias=False);self.source=nn.Parameter(torch.empty(heads,self.d));self.destination=nn.Parameter(torch.empty(heads,self.d));self.edge=nn.Linear(8,heads,bias=False);self.out=nn.Linear(dim,dim);self.norm=nn.LayerNorm(dim)
        nn.init.xavier_uniform_(self.source);nn.init.xavier_uniform_(self.destination)
    def forward(self,h,edge_index,attrs,gates):
        a,b=edge_index;v=self.project(h).reshape(len(h),self.heads,self.d)
        score=F.leaky_relu((v[a]*self.source).sum(-1)+(v[b]*self.destination).sum(-1)+self.edge(attrs),.2)+torch.log(gates.clamp_min(.03))[:,None]
        index=b[:,None].expand(-1,self.heads);mx=torch.full((len(h),self.heads),-torch.inf,device=h.device);mx.scatter_reduce_(0,index,score.detach(),reduce='amax',include_self=True)
        weight=(score-mx[b]).exp();den=torch.zeros_like(mx).scatter_add_(0,index,weight);weight=weight/den[b].clamp_min(1e-8)
        agg=torch.zeros_like(v);agg.index_add_(0,b,F.dropout(weight,.05,self.training)[:,:,None]*v[a])
        return self.norm(h+F.elu(self.out(agg.reshape(len(h),-1))))

class Model(nn.Module):
    def __init__(self,dim,kind):
        super().__init__();self.kind=kind
        self.encoder=nn.Sequential(nn.Linear(dim,64),nn.LayerNorm(64),nn.ELU(),nn.Linear(64,64),nn.ELU())
        self.edge_head=nn.Sequential(nn.Linear(136,64),nn.ELU(),nn.Linear(64,2))
        self.layers=nn.ModuleList([Layer(),Layer()]) if kind=='gat' else nn.ModuleList()
        self.pixel=nn.Sequential(nn.Linear(69,48),nn.ELU(),nn.Linear(48,3))
        # Exact baseline output at initialization. Every added parameter is newly initialized.
        nn.init.zeros_(self.pixel[-1].weight);nn.init.zeros_(self.pixel[-1].bias)
    def forward(self,x,pairs,attrs):
        h=self.encoder(x);a,b=pairs.T;symmetric=attrs.clone();symmetric[:,1:3]=symmetric[:,1:3].abs()
        edge=self.edge_head(torch.cat([(h[a]-h[b]).abs(),h[a]*h[b],symmetric],1))
        connection=edge[:,0];boundary=edge[:,1]
        # Same biological root may change class at its growing tip; never turn class sameness into identity.
        gates=(.05+.95*connection.sigmoid())*(1-.6*boundary.sigmoid())
        n=torch.arange(len(h),device=h.device);edge_index=torch.cat([pairs.T,pairs.flip(1).T,torch.stack([n,n])],1)
        reverse=attrs.clone();reverse[:,1:3]*=-1
        aa=torch.cat([attrs,reverse,attrs.new_zeros((len(h),8))]);gg=torch.cat([gates,gates,gates.new_ones(len(h))])
        for layer in self.layers:h=layer(h,edge_index,aa,gg)
        return h,connection,boundary
    def logits(self,h,node,pixel_features):
        residual=3*torch.tanh(self.pixel(torch.cat([h[node],pixel_features],1)))
        return pixel_features[:,:3]+residual
