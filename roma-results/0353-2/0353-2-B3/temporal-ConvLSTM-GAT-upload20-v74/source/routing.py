"""Biological continuation is an endpoint matching problem, not KNN components."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from prepare import GEO,js

def build_routing(folder,rawgraph,model,node_split,root_target):
    graph=dict(np.load(folder/'graph.npz'));N=len(graph['x']);pairs=graph['pairs'];attrs=graph['attrs'];phase=graph['phase'];centers=graph['centers'];raw=rawgraph['features'];parent=np.arange(N)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for a,b in pairs[attrs[:,5]>.5]:
        aa,bb=find(int(a)),find(int(b))
        if aa!=bb:parent[bb]=aa
    _,native=np.unique([find(i) for i in range(N)],return_inverse=True)
    is_junction=np.zeros(N,bool);owners=dict(np.load(folder/'ownership.npz'))
    for ti,t in enumerate(['T1','T2','T3','T4']):
        sk=skeletonize(owners[t]>=0);degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
        ids=np.flatnonzero((phase==ti)&(raw[:,3]<1e-6));xy=np.rint(centers[ids]).astype(int);is_junction[ids]=degree[xy[:,1],xy[:,0]]>=3
    endnode=[];endpos=[];endout=[];endchain=[];endphase=[]
    for chain in np.unique(native[~is_junction]):
        nodes=np.flatnonzero((native==chain)&~is_junction);nodes.sort()
        if not len(nodes):continue
        if len(nodes)==1:
            n=int(nodes[0]);v=raw[n,4:6].astype(float);v/=max(np.linalg.norm(v),1e-8);directions=[-v,v];chosen=[n,n]
        else:
            chosen=[int(nodes[0]),int(nodes[-1])];directions=[centers[nodes[0]]-centers[nodes[1]],centers[nodes[-1]]-centers[nodes[-2]]];directions=[v/max(np.linalg.norm(v),1e-8) for v in directions]
        for n,v in zip(chosen,directions):
            endnode.append(n);endout.append(v);endpos.append(centers[n]+v*float(raw[n,2])*6);endchain.append(chain);endphase.append(int(phase[n]))
    endnode=np.array(endnode,np.int64);endpos=np.array(endpos);endout=np.array(endout);endchain=np.array(endchain);endphase=np.array(endphase)
    bridges=[];extra=set()
    for ti in range(4):
        ids=np.flatnonzero(endphase==ti);tree=cKDTree(endpos[ids])
        for i,j in sorted(tree.query_pairs(48)):
            aa,bb=int(ids[i]),int(ids[j]);a,b=int(endnode[aa]),int(endnode[bb])
            if endchain[aa]==endchain[bb] or a==b:continue
            delta=endpos[bb]-endpos[aa];dist=np.linalg.norm(delta);unit=delta/max(dist,1e-8);fa=float(endout[aa]@unit);fb=float(endout[bb]@(-unit));alignment=float(-endout[aa]@endout[bb])
            if dist>4 and (fa<-.4 or fb<-.4):continue
            if alignment<-.35:continue
            key=tuple(sorted((a,b)));extra.add(key);bridges.append((aa,bb,key,dist,alignment,fa+fb))
    oldattrs={tuple(p):a for p,a in zip(pairs,attrs)};newpairs=np.array(sorted(set(oldattrs)|extra),np.int64);newattrs=[]
    for a,b in newpairs:
        if (a,b) in oldattrs:newattrs.append(oldattrs[(a,b)]);continue
        delta=centers[b]-centers[a];distance=np.linalg.norm(delta)
        newattrs.append([distance/32,delta[0]/32,delta[1]/32,abs(float(raw[a,4:6]@raw[b,4:6])),abs(int(phase[a]-phase[b])),0,abs(float(raw[a,6]-raw[b,6])),min(distance,100)/100])
    newattrs=np.array(newattrs,np.float32);a,b=newpairs.T;known=(root_target[a]>=0)&(root_target[b]>=0)&(newattrs[:,5]<.5);target=np.full(len(newpairs),-1,np.int64);target[known]=(root_target[a[known]]==root_target[b[known]]).astype(np.int64);split=np.full(len(newpairs),-1,np.int8)
    for k in [0,1,2]:split[known&(node_split[a]==k)&(node_split[b]==k)]=k
    index={tuple(p):i for i,p in enumerate(newpairs)};route_index=np.array([index[v[2]] for v in bridges],np.int64)
    graph.update(pairs=newpairs,attrs=newattrs,target=target,split=split);np.savez_compressed(folder/'graph.npz',**graph)
    np.savez_compressed(folder/'routing.npz',native_group=native,is_junction=is_junction,end_node=endnode,end_chain=endchain,bridge_a=np.array([v[0] for v in bridges],np.int64),bridge_b=np.array([v[1] for v in bridges],np.int64),bridge_pair_index=route_index,bridge_distance=np.array([v[3] for v in bridges],np.float32),bridge_alignment=np.array([v[4] for v in bridges],np.float32),bridge_facing=np.array([v[5] for v in bridges],np.float32))
    js(folder/'routing_verification.json',dict(native_chains=int(native.max()+1),junction_nodes=int(is_junction.sum()),endpoints=len(endnode),endpoint_candidates=len(bridges),added_graph_pairs=len(newpairs)-len(pairs),ground_truth_used_in_candidates=False,each_endpoint_maximum_one_continuation=True,junction_pixel_assignment='Single root, without unioning incident arms'))
