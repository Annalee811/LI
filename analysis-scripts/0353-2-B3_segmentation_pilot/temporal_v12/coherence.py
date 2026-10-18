"""Prediction-only skeleton-chain decoding. Junctions never imply root identity."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

def graph(mask):
    sk=skeletonize(mask);nodes=np.argwhere(sk);lookup={tuple(p):i for i,p in enumerate(nodes)};adj=[[] for _ in nodes]
    for i,(y,x) in enumerate(nodes):
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            j=lookup.get((y+dy,x+dx))
            if j is None:continue
            if dy and dx and ((y+dy,x) in lookup or (y,x+dx) in lookup):continue
            adj[i].append(j)
    return sk,nodes,adj

def chains(mask):
    sk,nodes,adj=graph(mask);seen=set();paths=[]
    starts=[i for i,a in enumerate(adj) if len(a)!=2]+list(range(len(nodes)))
    for start in starts:
        for nxt in adj[start]:
            edge=tuple(sorted((start,nxt)))
            if edge in seen:continue
            seen.add(edge);path=[start,nxt];prev,cur=start,nxt
            while len(adj[cur])==2:
                nxt=next(v for v in adj[cur] if v!=prev);edge=tuple(sorted((cur,nxt)))
                if edge in seen:break
                seen.add(edge);path.append(nxt);prev,cur=cur,nxt
            # Branch/crossing nodes are not assigned to either incident chain.
            if len(adj[path[0]])>=3:path=path[1:]
            if path and len(adj[path[-1]])>=3:path=path[:-1]
            if len(path)>=4 and path[0]!=path[-1]:paths.append(np.asarray(path,dtype=np.int64))
    return sk,nodes,paths,adj

def viterbi(emission,penalty):
    n=len(emission);score=-emission[0];back=np.zeros((n,3),np.uint8);cost=penalty*(1-np.eye(3))
    for i in range(1,n):
        mat=score[:,None]+cost;back[i]=mat.argmin(0);score=mat.min(0)-emission[i]
    states=np.empty(n,np.uint8);states[-1]=score.argmin()
    for i in range(n-1,0,-1):states[i-1]=back[i,states[i]]
    return states

def decode(probability,mask,penalty=0):
    pred=np.where(mask,probability.argmax(0)+1,0).astype(np.uint8)
    if not penalty or not mask.any():return pred
    sk,nodes,paths,adj=chains(mask)
    if not len(nodes):return pred
    _,nearest=ndi.distance_transform_edt(~sk,return_indices=True)
    ids=np.full(mask.shape,-1,np.int32);ids[tuple(nodes.T)]=np.arange(len(nodes));owner=ids[tuple(nearest)]
    on=owner[mask];logs=np.log(np.maximum(probability[:,mask],1e-7));counts=np.bincount(on,minlength=len(nodes))
    # Width-normalized evidence: thick roots do not overwhelm fine-root paths.
    emission=np.stack([np.bincount(on,weights=v,minlength=len(nodes))/np.maximum(counts,1) for v in logs],1)
    labels=np.full(len(nodes),-1,np.int16)
    for path in paths:labels[path]=viterbi(emission[path],penalty)+1
    component,_=ndi.label(mask,structure=np.ones((3,3),bool))
    same_component=component==component[tuple(nearest)]
    junctions=np.zeros(mask.shape,bool)
    junction_nodes=[i for i,a in enumerate(adj) if len(a)>=3]
    if junction_nodes:junctions[tuple(nodes[junction_nodes].T)]=True
    protected=ndi.binary_dilation(junctions,iterations=2)
    replace=mask&(labels[owner]>0)&same_component&~protected;pred[replace]=labels[owner[replace]]
    assert np.array_equal(pred>0,mask)
    return pred

def fragmentation(pred,mask):
    _,nodes,paths,_=chains(mask);switches=edges=short_runs=0
    for path in paths:
        vals=pred[tuple(nodes[path].T)];change=np.flatnonzero(vals[1:]!=vals[:-1])+1
        runs=np.diff(np.r_[0,change,len(vals)]);switches+=len(change);edges+=len(vals)-1;short_runs+=int((runs<5).sum())
    return dict(chain_count=len(paths),chain_edges=edges,class_switches=switches,switches_per_1000_edges=1000*switches/max(1,edges),runs_shorter_than_5px=short_runs)

def path_loss(logits,gt):
    """Train only same-class human skeleton runs; never smooth true class changes."""
    import torch
    terms=[];prob=logits.softmax(1)
    for b,yy in enumerate(gt.detach().cpu().numpy()):
        _,nodes,paths,_=chains((yy>0)&(yy<255));used=0
        for path in paths:
            xy=nodes[path];values=yy[tuple(xy.T)];bounds=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)]
            for lo,hi in zip(bounds[:-1],bounds[1:]):
                if hi-lo<8:continue
                ix=xy[lo:hi:2];p=prob[b,:,ix[:,0],ix[:,1]];label=int(values[lo])-1
                terms.append(-torch.log(p[label].mean().clamp_min(1e-6))+.5*(p-p.mean(1,keepdim=True)).square().sum(0).mean());used+=1
                if used>=32:break
            if used>=32:break
    return torch.stack(terms).mean() if terms else logits.sum()*0
