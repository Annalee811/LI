"""Latest20 supervision, immutable original baseline, causal multi-date inputs."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/B3_ConvLSTM_GAT_upload20_v74_20261028'
BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028'
GEO=ROOT/'results/B3_temporal_GAT_new_growth_v68_20261028/data'
UPLOAD=Path('/workspace/attachments/5d98e0e1-a67b-46ab-b7e9-3983a62901aa/root_annotation_progress (20).json')
MODELS=['unet','deeplab','segformer'];PERIODS=['T1','T2','T3','T4'];SEED=2026102804
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvout(p,v):
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(v[0]));w.writeheader();w.writerows(v)
def unpack(v,shape,dtype=np.uint8):
    assert len(v)%2==0 and sum(v[1::2])==np.prod(shape)
    return np.repeat(np.array(v[::2],dtype),np.array(v[1::2],np.int64)).reshape(shape)
def topology_features(mask):
    sk=skeletonize(mask);degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
    prox=lambda p:np.exp(-ndi.distance_transform_edt(~p)**2/128).astype(np.float32) if p.any() else np.zeros(mask.shape,np.float32)
    return prox(sk&(degree==1)),prox(sk&(degree>=3)),np.minimum(ndi.distance_transform_edt(mask),8).astype(np.float32)/8
def resize(a,size,nearest=False):return np.array(Image.fromarray(a).resize(size,Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR))

def main():
    sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
    assert require_active_position('0353-2-B3')=='0353-2-B3';assert not OUT.exists(),'Never overwrite a run'
    OUT.mkdir();(OUT/'data').mkdir();(OUT/'source').mkdir();(OUT/'baseline_original').mkdir()
    shutil.copy2(UPLOAD,OUT/'data/source_progress_20.json');doc=json.loads(UPLOAD.read_text());assert sha(UPLOAD)=='73cc3061f5eda257ddd629bf545776615a688846d1f5b9fb5314e0de237954b8'
    st=doc['root_structure_annotation'];confirmed={h['root_id']:h for h in st['root_track_hierarchy'] if h.get('annotation_status_v60')=='user_defined'};lookup={rid:i for i,rid in enumerate(sorted(confirmed))}
    rgb={};gt={};rootmask={};identity={};earliest={};audits=[];immutable=[dict(path=str(OUT/'data/source_progress_20.json'),sha256=sha(UPLOAD))]
    (OUT/'data/reference').mkdir()
    for ti,t in enumerate(PERIODS):
        e=next(e for e in doc['images'] if e['name']==t+'_RGB.png');p=BASE/'data/upper'/e['name'];assert sha(p)==e['image_sha256'];rgb[t]=np.array(Image.open(p)).astype(np.float32)/255
        mask=unpack(e['layers']['mask'],(1108,2541));tmp=unpack(e['layers']['temporal'],mask.shape);rootmask[t]=mask==255
        y=np.zeros(mask.shape,np.uint8);y[(mask==64)|(mask==128)|((mask==255)&(tmp==0))]=255
        for old,new in [(3,1),(1,2),(2,3),(4,4)]:y[(mask==255)&(tmp==old)]=new
        gt[t]=y;Image.fromarray(y).save(OUT/'data/reference'/f'{t}_classes.png');Image.fromarray(mask).save(OUT/'data/reference'/f'{t}_mask.png')
        f=next(f for f in st['periods'] if f['period']==t);owner=unpack(f['owner_map_rle'],mask.shape,np.int32);lut=np.full(int(owner.max())+1,-1,np.int32)
        for s in f['segments']:
            if s['root_id'] in lookup:lut[s['index']]=lookup[s['root_id']]
        ident=lut[owner];ident[~rootmask[t]]=-1;identity[t]=ident
        for rid in np.unique(ident[ident>=0]):earliest.setdefault(int(rid),ti)
        audits.append(dict(period=t,typed_pixels={str(k):int((y==k).sum()) for k in [1,2,3,4,255]},new_root_pixels_without_owner=int((rootmask[t]&(owner==0)).sum()),stale_owner_pixels_not_supervised=int(((owner>0)&~rootmask[t]).sum())))
        immutable.append(dict(path=str(p),sha256=sha(p)))
    spatial=np.full((1108,2541),-1,np.int8)
    for r in rows(BASE/'data/spatial_manifest.csv'):
        x,y,w,h=[int(r[k]) for k in ['x','y','width','height']];sl=['train','val','test'].index(r['split']);assert ((spatial[y:y+h,x:x+w]<0)|(spatial[y:y+h,x:x+w]==sl)).all();spatial[y:y+h,x:x+w]=sl
    np.save(OUT/'data/spatial.npy',spatial)
    for m in MODELS:
        d=OUT/'data'/m;d.mkdir();old=OUT/'baseline_original'/m;old.mkdir();
        for name in ['history.csv','config.json','completion.json','best_temporal_head.pt']:
            shutil.copy2(BASE/m/name,old/name);immutable.append(dict(path=str(BASE/m/name),sha256=sha(BASE/m/name)))
        (old/'full').symlink_to((BASE/m/'full').resolve(),target_is_directory=True)
        graph=dict(np.load(GEO/m/'graph.npz'));maps=dict(np.load(GEO/m/'ownership.npz'));N=len(graph['features']);ns=graph['node_split'];pair=graph['pairs'];a,b=pair.T
        allframes=[];pixeldata={};hist=np.zeros((N,3),np.int64);node_rid=np.full(N,-1,np.int32);region_counts=[];rng=np.random.default_rng(SEED);samples={0:[],1:[],2:[]};base_stats=np.zeros((N,7),np.float32)
        gates={t:maps[t]>=0 for t in PERIODS};topo={t:topology_features(gates[t]) for t in PERIODS}
        for ti,t in enumerate(PERIODS):
            owner=maps[t];flat=np.flatnonzero(gates[t]);nid=owner.ravel()[flat];yy,xx=np.divmod(flat,2541);prev=PERIODS[max(0,ti-1)];tp,jp,wd=topo[t]
            ppath=Path(json.loads((BASE/'preparation.json').read_text())['roots'])/m/'upper'/f'{t}_probability.npy';rootprob=np.load(ppath)
            raw=np.concatenate([rgb[t],rootprob[:,:,None],gates[t][:,:,None].astype(np.float32),wd[:,:,None],tp[:,:,None],jp[:,:,None]],2)
            frame=np.stack([resize(raw[:,:,c],(318,139)) for c in range(8)],0);allframes.append(frame.astype(np.float16))
            if ti:
                qpath=BASE/m/'full'/f'{t}_probability.npz';q=np.load(qpath)['probability'].astype(np.float32);q/=np.maximum(q.sum(0),1e-7);immutable.append(dict(path=str(qpath),sha256=sha(qpath)))
            else:q=np.full((3,1108,2541),1/3,np.float32)
            probs=q.reshape(3,-1).T[flat];counts=np.bincount(nid,minlength=N);means=np.stack([np.bincount(nid,weights=probs[:,k],minlength=N)/np.maximum(counts,1) for k in range(3)],1);var=np.stack([np.bincount(nid,weights=probs[:,k]**2,minlength=N)/np.maximum(counts,1) for k in range(3)],1)-means**2;ids=np.flatnonzero(graph['phase']==ti);base_stats[ids,:3]=means[ids];base_stats[ids,3:6]=np.sqrt(np.maximum(var[ids],0));base_stats[ids,6]=-(means[ids]*np.log(np.maximum(means[ids],1e-7))).sum(1)
            offset=np.c_[xx-graph['centers'][nid,0],yy-graph['centers'][nid,1]]/12
            pd=np.minimum(ndi.distance_transform_edt(~gates[prev]),32).astype(np.float32)/32
            pf=np.c_[np.log(np.maximum(probs,1e-5)),offset,rgb[t].reshape(-1,3)[flat],np.abs(rgb[t]-rgb[prev]).reshape(-1,3)[flat],pd.ravel()[flat],topo[prev][0].ravel()[flat],tp.ravel()[flat],jp.ravel()[flat]].astype(np.float32)
            target=gt[t].ravel()[flat].astype(np.int16)-1;valid=(target>=0)&(target<3);split=np.where(valid,spatial.ravel()[flat],-1);rid=identity[t].ravel()[flat]
            pixeldata[t]=dict(pixel=flat,node=nid,pf=pf,target=target,split=split,rid=rid);np.savez_compressed(d/(t+'_pixels.npz'),**pixeldata[t])
            # Identity supervision is taken only from each node's own spatial mask.
            sort=np.argsort(nid,kind='stable');c=np.bincount(nid,minlength=N);ends=np.cumsum(c);starts=np.r_[0,ends[:-1]]
            for n in ids:
                chosen=sort[starts[n]:ends[n]];label=rid[chosen];ss=spatial.ravel()[flat[chosen]];label=label[ss==ns[n]];known=label[label>=0]
                if len(known)>=3:
                    ct=np.bincount(known,minlength=len(lookup))
                    if ct.max()/len(known)>=.9 and len(known)/max(len(label),1)>=.6:node_rid[n]=ct.argmax()
            if not ti:continue
            train=np.flatnonzero(split==0);np.add.at(hist,(nid[train],target[train]),1)
            chosen=[]
            for k in range(3):
                ix=np.flatnonzero((split==0)&(target==k));order=ix[np.argsort(nid[ix],kind='stable')]
                if not len(order):continue
                cuts=np.r_[0,np.flatnonzero(nid[order][1:]!=nid[order][:-1])+1,len(order)]
                for lo,hi in zip(cuts[:-1],cuts[1:]):
                    s=order[lo:hi];chosen.extend(s if len(s)<=8 else rng.choice(s,8,replace=False))
            for k in [0,1,2]:
                ix=np.array(chosen,np.int64) if k==0 else np.flatnonzero(split==k)
                samples[k].append(dict(node=nid[ix],pf=pf[ix],target=target[ix].astype(np.int64),phase=np.full(len(ix),ti,np.int64),pixel=flat[ix]))
            immutable.extend([dict(path=str(ppath),sha256=sha(ppath)),dict(path=str(GEO/m/'ownership.npz'),sha256=sha(GEO/m/'ownership.npz'))])
        # Recompute all connection targets from20; no17/18 supervision is reused.
        conn=np.full(len(pair),-1,np.int64);known=(node_rid[a]>=0)&(node_rid[b]>=0)&(graph['attrs'][:,5]<.5);conn[known]=(node_rid[a[known]]==node_rid[b[known]]).astype(np.int64);cs=np.full(len(pair),-1,np.int8)
        for k in [0,1,2]:cs[known&(ns[a]==k)&(ns[b]==k)]=k
        samephase=graph['phase'][a]==graph['phase'][b]
        for k in [0,1,2]:
            for phase in [samephase,~samephase]:assert np.all(np.bincount(conn[(cs==k)&phase],minlength=2)>0)
        # Root-period state supervision: only high-purity known-root cases in training pixels.
        groups=[];group_nodes=[];group_target=[];conflicts=[]
        for ti,t in enumerate(PERIODS[1:],1):
            px=pixeldata[t]
            for rid in range(len(lookup)):
                take=(px['split']==0)&(px['rid']==rid);ys=px['target'][take]
                if len(ys)<8:continue
                ratio=float((ys==1).mean());state=1 if ratio>=.98 else 0 if ratio<=.02 else -1
                if state==1 and earliest.get(rid,ti)<ti:
                    conflicts.append(dict(period=t,root_id=list(sorted(lookup))[rid],new_fraction=ratio,pixels=len(ys),reason='Earlier confirmed identity already visible; root state is ambiguous'));continue
                if state<0:conflicts.append(dict(period=t,root_id=list(sorted(lookup))[rid],new_fraction=ratio,pixels=len(ys)));continue
                ids=np.unique(px['node'][take]);groups.append(dict(period=t,root_target=rid,node_ids=ids.tolist(),state_new=state,pixels=len(ys)));group_nodes.extend(ids.tolist());group_target.append(state)
        assert group_target and len(set(group_target))==2,'Need both clean new and established roots'
        ptr=np.r_[0,np.cumsum([len(v['node_ids']) for v in groups])];np.savez_compressed(d/'root_state.npz',nodes=np.array(group_nodes,np.int64),ptr=ptr,target=np.array(group_target,np.int64));js(d/'root_state_audit.json',dict(groups=groups,ambiguous_training_cases_excluded_from_root_state_loss=conflicts,labels_not_modified=True))
        for k,name in enumerate(['train','val','test']):
            parts=samples[k];np.savez_compressed(d/(name+'.npz'),**{key:np.concatenate([p[key] for p in parts]) for key in parts[0]})
        raw=np.c_[graph['features'],base_stats];mu=raw[ns==0].mean(0);sd=raw[ns==0].std(0);sd[sd<1e-4]=1
        np.savez_compressed(d/'graph.npz',x=((raw-mu)/sd).astype(np.float32),pairs=pair,attrs=graph['attrs'],phase=graph['phase'],centers=graph['centers'],target=conn,split=cs,node_split=ns,root_target=node_rid,train_class_hist=hist)
        np.savez_compressed(d/'frames.npz',frames=np.stack(allframes));np.savez_compressed(d/'ownership.npz',**maps)
        js(d/'preparation.json',dict(model=m,nodes=N,pairs=len(pair),frame_shape=[4,8,139,318],pixel_features=15,clean_root_states=dict(Counter(group_target)),ambiguous_root_state_cases=len(conflicts),class_counts={s:np.bincount(np.load(d/(s+'.npz'))['target'],minlength=3).tolist() for s in ['train','val','test']}))
        from routing import build_routing
        build_routing(d,graph,m,ns,node_rid)
        print('PREPARED',m,'root states',dict(Counter(group_target)),'conflicts',len(conflicts),flush=True)
    # Historical losses retain their original reference and objective; no refitting/curve filling.
    shutil.copytree(BASE/'figures',OUT/'baseline_original/training_figures')
    js(OUT/'preparation.json',dict(position='0353-2-B3',upload20_sha256=sha(UPLOAD),confirmed_root_IDs=len(lookup),annotation_revision=doc.get('annotation_revision'),periods=audits,epochs=100,new_variants=['control','convlstm_gat'],causal_history=True,original_reference='V66 archived18 baseline, preserved unchanged',comparison_reference='Latest20 for all variants; same frozen binary supports',root_state_ambiguity='Preserve20 exactly; exclude mixed cases only from root-state supervision, not pixel evaluation',immutable_inputs=list({e['path']:e for e in immutable}.values()),outer_partition='All4 B3 dates in one development position; inner spatial masks only, no independent test'))
    for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
    (OUT/'README.md').write_text('''# Upload20 ConvLSTM + GAT temporal optimization

All targets come only from latest20, preserved byte-for-byte. Reference contains
U-Net-assisted revision metadata; agreement is internal, not independent accuracy.
Frozen binary supports/widths/lengths are inherited unchanged. Pixel edits without
manual ownership can supervise temporal pixels but are not assigned invented IDs.

Historical V66 predictions/weights/real100-epoch curves are preserved in
baseline_original. Those curves were trained on18; their labels/objectives are
never relabeled as20. All comparison metrics are recomputed against20.

Three newly initialized ConvLSTM+GAT refiners and three matched updated-label
controls,100 actual full-graph/sequence epochs each. No old GAT/ConvLSTM weights.
ConvLSTM is a real2D recurrent cell, processing T1->T4; GAT edges allow same-date
messages and past->future only. No future date is used to classify an earlier one.
GAT learns same-root links after graph propagation, with within-date and cross-date
losses balanced separately. Root state is learned, not majority-vote recoloring.
New roots and established roots are decoded as separate states; only established
roots have persistent/tip classes. Root IDs at inference are predicted groups,
never ground-truth IDs. Biological fragmentation/mixture errors remain evaluated.

Ambiguous root-state targets are audited and excluded only from state loss, while
all valid latest20 reference pixels remain in metrics. No human label is changed.
Checkpoints save every real epoch, optimizer/scheduler/RNG; partial epochs replay.
STOPPED_BY_USER.json blocks recovery. Same B3 outer development field, fixed inner
spatial masks; no external independent tube/position generalization is claimed.

Requested outputs: archived initial four-period/local/extension figures and full
100 epochs; optimized four-period matched and high-disagreement details (including
regressions); three-class P/R/Dice/IoU, connection and temporal/identity continuity
tables. Difference windows are illustrative, never selected using test accuracy.
''')
    print('DATA_COMPLETE',OUT,flush=True)
if __name__=='__main__':main()
