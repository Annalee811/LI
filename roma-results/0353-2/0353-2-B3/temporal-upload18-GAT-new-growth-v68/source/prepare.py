"""Prediction-only native graphs; upload18 supervises temporal classes, not identity."""
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
OUT=ROOT/'results/B3_temporal_GAT_new_growth_v68_20261028'
PERIODS=['T1','T2','T3','T4'];MODELS=['unet','deeplab','segformer'];SEED=2026102802
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(p,v):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)

def native_chunks(mask):
    """Cut skeleton at every branch/crossing; do not guess which crossing arms join."""
    sk=skeletonize(mask);yx=np.argwhere(sk);lookup={tuple(p):i for i,p in enumerate(yx)};adj=[[] for _ in yx]
    for i,(y,x) in enumerate(yx):
        for dy,dx in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            j=lookup.get((y+dy,x+dx))
            if j is None:continue
            if dy and dx and ((y+dy,x) in lookup or (y,x+dx) in lookup):continue
            adj[i].append(j)
    seen=set();paths=[]
    for start in [i for i,a in enumerate(adj) if len(a)!=2]+list(range(len(yx))):
        for nxt in adj[start]:
            edge=tuple(sorted((start,nxt)))
            if edge in seen:continue
            seen.add(edge);path=[start,nxt];prev,cur=start,nxt
            while len(adj[cur])==2:
                nxt=next(v for v in adj[cur] if v!=prev);edge=tuple(sorted((cur,nxt)))
                if edge in seen:break
                seen.add(edge);path.append(nxt);prev,cur=cur,nxt
            # Crossings have separate singleton nodes, never arbitrarily assigned to one incident root.
            path=[i for i in path if len(adj[i])<3]
            if path:paths.append(path)
    assigned=np.full(len(yx),-1,np.int32);chunks=[];chain_ids=[];chain=0
    for path in paths:
        path=[i for i in path if assigned[i]<0]
        for lo in range(0,len(path),12):
            ids=path[lo:lo+12]
            if not ids:continue
            assigned[ids]=len(chunks);chunks.append(yx[ids][:,::-1]);chain_ids.append(chain)
        chain+=1
    for i in np.flatnonzero(assigned<0):
        assigned[i]=len(chunks);chunks.append(yx[i:i+1,::-1]);chain_ids.append(chain);chain+=1
    at=np.full(mask.shape,-1,np.int32);at[tuple(yx.T)]=assigned
    _,nearest=ndi.distance_transform_edt(~sk,return_indices=True);owner=at[tuple(nearest)];owner[~mask]=-1
    component,_=ndi.label(mask,structure=np.ones((3,3),bool))
    bad=mask&(component!=component[tuple(nearest)])
    for cl in np.unique(component[bad]):
        take=np.flatnonzero(component[tuple(yx.T)]==cl);by,bx=np.where(bad&(component==cl));near=cKDTree(yx[take]).query(np.c_[by,bx])[1];owner[by,bx]=assigned[take[near]]
    assert (owner[mask]>=0).all()
    return chunks,np.array(chain_ids),owner

def build_geometry(m):
    data=OUT/'data'/m;data.mkdir(parents=True,exist_ok=True)
    if (data/'geometry_complete.json').exists():return
    prep=json.loads((BASE/'preparation.json').read_text());roots=Path(prep['roots'])/m/'upper'
    split=np.full((1108,2541),-1,np.int8)
    for r in rows(BASE/'data/spatial_manifest.csv'):
        x,y,w,h=[int(r[k]) for k in ['x','y','width','height']];sl=['train','val','test'].index(r['split'])
        assert ((split[y:y+h,x:x+w]<0)|(split[y:y+h,x:x+w]==sl)).all();split[y:y+h,x:x+w]=sl
    allpoints=[];features=[];centers=[];tangents=[];phase=[];chainlist=[];owners={};pixeldata={};node_splits=[];native_edges=[]
    immutable=[];tic=time.monotonic()
    rgb={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')).astype(np.float32)/255 for t in PERIODS}
    masks={t:np.array(Image.open(roots/f'{t}_root_mask.png'))>0 for t in PERIODS}
    for ti,t in enumerate(PERIODS):
        chunks,chains,owner=native_chunks(masks[t]);offset=len(allpoints)
        owner[masks[t]]+=offset;owners[t]=owner
        gt=np.array(Image.open(BASE/'data/current'/f'{t}_classes.png'))
        flat=np.flatnonzero(masks[t]);nid=owner.ravel()[flat];yy,xx=np.divmod(flat,2541)
        order=np.argsort(nid,kind='stable');ordered=flat[order];counts=np.bincount(nid-offset,minlength=len(chunks));ends=np.cumsum(counts);starts=np.r_[0,ends[:-1]]
        prev=PERIODS[max(0,ti-1)];prevdist=ndi.distance_transform_edt(~masks[prev]);curwidth=ndi.distance_transform_edt(masks[t])
        localfeatures=[]
        for j,q in enumerate(chunks):
            n=offset+j;inds=ordered[starts[j]:ends[j]];pix=rgb[t].reshape(-1,3)[inds]
            pp=rgb[prev].reshape(-1,3)[inds];p=q.astype(float);center=p.mean(0);vec=p[-1]-p[0];length=np.linalg.norm(vec);vec=vec/max(length,1)
            if length==0:vec=np.array([1.,0.])
            arc=np.linalg.norm(np.diff(p,axis=0),axis=1).sum();width=float(curwidth.ravel()[inds].mean()) if len(inds) else 1
            feat=[center[0]/2541,center[1]/1108,length/12,arc/16,vec[0],vec[1],width/10,
                float(prevdist.ravel()[inds].mean())/32 if len(inds) else 1,*(pix.mean(0) if len(pix) else [0,0,0]),
                *(pix.std(0) if len(pix) else [0,0,0]),*((pix-pp).mean(0) if len(pix) else [0,0,0]),*np.eye(4)[ti]]
            localfeatures.append(feat);centers.append(center);tangents.append(vec);allpoints.append(q);phase.append(ti);chainlist.append(int(chains[j]));
            ss=split.ravel()[inds];cc=np.bincount(ss[ss>=0],minlength=3);node_splits.append(int(cc.argmax()) if cc.sum() else -1)
        features.extend(localfeatures)
        # Every frozen root pixel has exactly one owner, even at crossings; no pixel is added or removed.
        assert np.array_equal(owner>=0,masks[t])
        basepixel=np.c_[xx-np.array(centers)[nid,0],yy-np.array(centers)[nid,1]]/12
        label=gt.ravel()[flat].astype(np.int16)-1;valid=(label>=0)&(label<3)
        pixeldata[t]=dict(pixel=flat,node=nid,xy_offset=basepixel.astype(np.float32),target=label,split=np.where(valid,split.ravel()[flat],-1))
        bychain={}
        for j,ch in enumerate(chains):bychain.setdefault(ch,[]).append(offset+j)
        for ids in bychain.values():native_edges.extend(zip(ids[:-1],ids[1:]))
        for path in [roots/f'{t}_root_mask.png',BASE/'data/upper'/f'{t}_RGB.png',BASE/'data/current'/f'{t}_classes.png']:
            immutable.append(dict(path=str(path),sha256=sha(path)))
        print('GEOMETRY',m,t,'nodes',len(chunks),'root pixels',len(flat),'sec',round(time.monotonic()-tic,1),flush=True)
    centers=np.array(centers);tangents=np.array(tangents);phase=np.array(phase);chainlist=np.array(chainlist);N=len(features)
    pairs={tuple(sorted((int(a),int(b)))) for a,b in native_edges}
    for ti in range(4):
        ids=np.flatnonzero(phase==ti);tree=cKDTree(centers[ids]);dd,qq=tree.query(centers[ids],k=min(7,len(ids)))
        for a,dist,bb in zip(ids,dd,qq):
            for d,j in zip(np.atleast_1d(dist),np.atleast_1d(bb)):
                if d<=26 and a!=ids[j]:pairs.add(tuple(sorted((int(a),int(ids[j])))))
        if ti:
            prev=np.flatnonzero(phase==ti-1);tree=cKDTree(centers[prev]);dd,qq=tree.query(centers[ids],k=min(4,len(prev)))
            for a,dist,bb in zip(ids,dd,qq):
                for d,j in zip(np.atleast_1d(dist),np.atleast_1d(bb)):
                    if d<=22:pairs.add(tuple(sorted((int(a),int(prev[j])))))
    pairs=np.array(sorted(pairs),np.int64);a,b=pairs.T;delta=centers[b]-centers[a];samepath=(phase[a]==phase[b])&(chainlist[a]==chainlist[b])
    attrs=np.c_[np.linalg.norm(delta,axis=1)/32,delta/32,np.abs((tangents[a]*tangents[b]).sum(1)),np.abs(phase[a]-phase[b]),samepath.astype(float),np.abs(np.array(features)[a,6]-np.array(features)[b,6]),np.minimum(np.linalg.norm(delta,axis=1),100)/100].astype(np.float32)
    # User explicitly authorized upload17's user-defined IDs as TARGETS ONLY.
    # Graph geometry remains predicted, and temporal targets still come only from18.
    source17=ROOT/'results/B3_temporal_gat_v65_20261027/data/source_progress_17.json'
    raw17=json.loads(source17.read_text());structure=raw17['root_structure_annotation']
    confirmed={h['root_id']:h for h in structure['root_track_hierarchy'] if h.get('annotation_status_v60')=='user_defined'}
    numbering={rid:i for i,rid in enumerate(sorted(confirmed))};root_target=np.full(N,-1,np.int32);root_coverage=np.zeros(N,np.float32);root_purity=np.zeros(N,np.float32)
    for t in PERIODS:
        frame=next(f for f in structure['periods'] if f['period']==t);rle=frame['owner_map_rle']
        owner17=np.repeat(np.array(rle[::2],np.int32),np.array(rle[1::2],np.int64)).reshape(1108,2541)
        lut=np.full(max(s['index'] for s in frame['segments'])+1,-1,np.int32)
        for s in frame['segments']:
            if s['root_id'] in numbering:lut[s['index']]=numbering[s['root_id']]
        rid=lut[owner17.ravel()[pixeldata[t]['pixel']]];ids=pixeldata[t]['node'];order=np.argsort(ids,kind='stable');counts=np.bincount(ids,minlength=N);bounds=np.cumsum(counts);beg=np.r_[0,bounds[:-1]]
        for n in np.flatnonzero(np.array(phase)==PERIODS.index(t)):
            pos=order[beg[n]:bounds[n]];labels=rid[pos];ss=split.ravel()[pixeldata[t]['pixel'][pos]];labels=labels[ss==node_splits[n]];known=labels[labels>=0]
            if not len(known):continue
            hist=np.bincount(known,minlength=len(confirmed));purity=hist.max()/len(known);coverage=len(known)/max(len(labels),1)
            root_coverage[n]=coverage;root_purity[n]=purity
            if coverage>=.6 and purity>=.9 and len(known)>=3:root_target[n]=hist.argmax()
    pair_target=np.full(len(pairs),-1,np.int8);known=(root_target[a]>=0)&(root_target[b]>=0)&~samepath
    pair_target[known]=(root_target[a[known]]==root_target[b[known]]).astype(np.int8)
    pair_split=np.full(len(pairs),-1,np.int8);node_splits=np.array(node_splits)
    for k in [0,1,2]:pair_split[known&(node_splits[a]==k)&(node_splits[b]==k)]=k
    np.savez_compressed(data/'graph.npz',pairs=pairs,attrs=attrs,features=np.array(features,np.float32),node_split=node_splits,phase=phase,centers=centers,tangents=tangents,pair_target=pair_target,pair_split=pair_split)
    np.savez_compressed(data/'identity_targets.npz',root_target=root_target,coverage=root_coverage,purity=root_purity)
    np.savez_compressed(data/'ownership.npz',**owners)
    for t in PERIODS:np.savez_compressed(data/(t+'_pixels.npz'),**pixeldata[t])
    immutable.append(dict(path=str(source17),sha256=sha(source17)))
    js(data/'geometry_complete.json',dict(model=m,nodes=N,edges=len(pairs),immutable_inputs=immutable,graph_inputs='RGB and frozen binary geometry only',human_root_IDs_input=False,connection_ground_truth='Only142 user_defined IDs in upload17; all assistant proposals and unassigned handles excluded',confirmed_IDs=len(confirmed),identity_lut=numbering,pair_counts={str(k):np.bincount(pair_target[pair_split==k],minlength=2).tolist() for k in [0,1,2]}))

def baseline_dir(m):
    full=BASE/m/'full'
    if all((full/f'{t}_probability.npz').exists() for t in PERIODS[1:]):return full
    if m=='unet':
        full=ROOT/'results/B3_temporal_upload18_UNet_preview_v67_20261028/unet/full'
        if all((full/f'{t}_probability.npz').exists() for t in PERIODS[1:]):return full
    return None

def add_baseline(m):
    data=OUT/'data'/m
    if (data/'preparation.json').exists():return True
    full=baseline_dir(m)
    if full is None or not (BASE/m/'completion.json').exists():return False
    geo=dict(np.load(data/'graph.npz'));N=len(geo['features']);mf=np.zeros((N,7),np.float32);sample_parts=[];eval_parts={1:[],2:[]};inputs=[];rng=np.random.default_rng(SEED)
    for ti,t in enumerate(PERIODS):
        px=dict(np.load(data/(t+'_pixels.npz')));ids=px['node'];allinds=np.flatnonzero(geo['phase']==ti)
        if ti==0:p=np.full((len(ids),3),1/3,np.float32)
        else:
            path=full/f'{t}_probability.npz';q=np.load(path)['probability'].astype(np.float32);q/=np.maximum(q.sum(0),1e-8);p=q.reshape(3,-1).T[px['pixel']];inputs.append(dict(path=str(path),sha256=sha(path)))
        count=np.bincount(ids,minlength=N);mean=np.stack([np.bincount(ids,weights=p[:,k],minlength=N)/np.maximum(count,1) for k in range(3)],1)
        var=np.stack([np.bincount(ids,weights=p[:,k]**2,minlength=N)/np.maximum(count,1) for k in range(3)],1)-mean**2
        mf[allinds,:3]=mean[allinds];mf[allinds,3:6]=np.sqrt(np.maximum(var[allinds],0));mf[allinds,6]=-(mean[allinds]*np.log(np.maximum(mean[allinds],1e-8))).sum(1)
        if ti==0:continue
        log=np.log(np.maximum(p,1e-5));pf=np.c_[log,px['xy_offset']].astype(np.float32)
        # Sample up to eight pixels per (node,class), including minority classes within a chunk.
        selected=[]
        for k in range(3):
            ix=np.flatnonzero((px['split']==0)&(px['target']==k));order=ix[np.argsort(ids[ix],kind='stable')]
            if not len(order):continue
            cuts=np.r_[0,np.flatnonzero(ids[order][1:]!=ids[order][:-1])+1,len(order)]
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                s=order[lo:hi];selected.extend(s if len(s)<=8 else rng.choice(s,8,replace=False))
        for split,parts in [(0,sample_parts),(1,eval_parts[1]),(2,eval_parts[2])]:
            ix=np.array(selected,np.int64) if split==0 else np.flatnonzero(px['split']==split)
            parts.append(dict(node=ids[ix],target=px['target'][ix].astype(np.int64),pixel_features=pf[ix],period=np.full(len(ix),ti),pixel=px['pixel'][ix]))
    def join(parts):return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for split,parts in [('train',sample_parts),('val',eval_parts[1]),('test',eval_parts[2])]:np.savez_compressed(data/(split+'.npz'),**join(parts))
    x=np.c_[geo['features'],mf];train=geo['node_split']==0;mu=x[train].mean(0);sd=x[train].std(0);sd[sd<1e-4]=1
    np.savez_compressed(data/'features.npz',x=((x-mu)/sd).astype(np.float32),mean=mu,std=sd)
    done=json.loads((BASE/m/'completion.json').read_text());assert done['epochs_completed']==100
    js(data/'preparation.json',dict(model=m,base_selected_epoch=done['selected_epoch'],base_checkpoint_sha256=done['checkpoint_sha256'],baseline_full=str(full),nodes=N,edges=len(geo['pairs']),input_features=x.shape[1],class_counts={s:np.bincount(np.load(data/(s+'.npz'))['target'],minlength=3).tolist() for s in ['train','val','test']},immutable_baselines=inputs,annotation_source='Temporal: upload18 only. Connection: user-confirmed IDs from upload17 only, explicitly authorized',identity_connection_supervision=True,uncertain_crossings='Unconfirmed/low-purity identity targets ignored; geometry candidates alone are not asserted true connections'))
    print('DATA_READY',m,flush=True);return True

if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'analysis-scripts'))
    from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3'
    OUT.mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    for m in MODELS:build_geometry(m);add_baseline(m)
