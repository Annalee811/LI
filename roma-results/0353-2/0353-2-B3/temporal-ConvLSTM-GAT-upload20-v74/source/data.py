import json
import numpy as np
import torch
from prepare import OUT

class Data:
    def __init__(self,model):
        self.model=model;self.folder=OUT/'data'/model;self.graph=dict(np.load(self.folder/'graph.npz'));self.routing=dict(np.load(self.folder/'routing.npz'))
        g=self.graph;self.x=torch.from_numpy(g['x']);self.frames=torch.from_numpy(np.load(self.folder/'frames.npz')['frames'].astype(np.float32));self.pairs=torch.from_numpy(g['pairs']);self.attrs=torch.from_numpy(g['attrs']);self.phase=torch.from_numpy(g['phase'].astype(np.int64));self.centers=torch.tensor(g['centers'],dtype=torch.float32)
        self.sets={s:dict(np.load(self.folder/(s+'.npz'))) for s in ['train','val','test']}
        self.tensor={s:{k:torch.from_numpy(v) for k,v in d.items() if k in ['node','pf','target']} for s,d in self.sets.items()}
        state=dict(np.load(self.folder/'root_state.npz'));self.state={k:torch.from_numpy(v) for k,v in state.items()}
        a,b=g['pairs'].T;self.same=g['phase'][a]==g['phase'][b];self.connection={k:np.flatnonzero(g['split']==k) for k in [0,1,2]}
        self.native_train=np.flatnonzero((g['attrs'][:,5]>.5)&(g['root_target'][a]>=0)&(g['root_target'][b]>=0)&(g['node_split'][a]==0)&(g['node_split'][b]==0))
        self.native_target=torch.tensor((g['root_target'][a[self.native_train]]==g['root_target'][b[self.native_train]]).astype(np.float32))
        hist=g['train_class_hist'];pure=(hist.sum(1)>=3)&(hist.max(1)/np.maximum(hist.sum(1),1)>=.9);cl=hist.argmax(1)
        self.stable_pairs=np.flatnonzero((g['split']==0)&~self.same&(g['target']==1)&pure[a]&pure[b]&(cl[a]==cl[b]))
        self.prep=json.loads((self.folder/'preparation.json').read_text())
        assert np.all(g['phase'][a]<=g['phase'][b]),'Stored chronological edge orientation invalid'
    def forward(self,model,frames=None,x=None):return model(self.frames if frames is None else frames,self.x if x is None else x,self.pairs,self.attrs,self.phase,self.centers)
