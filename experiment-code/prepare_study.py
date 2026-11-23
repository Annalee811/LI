from pathlib import Path
import numpy as np,json,hashlib,pickle,csv,sys,shutil
from PIL import Image
W=Path(__file__).resolve().parents[1];A=Path('/workspace/work/V90_all_microtubes_20261118/model_source/assets');R=Path('/workspace/work/V66_V90_same_field_SCI_20261113/sources/reference');sys.path.insert(0,'/workspace/LI/analysis-scripts');from dataset_identity import require_active_position,tube_id
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=dict(np.load(A/'local/data/unet/graph.npz'));N=len(g['x']);sp=np.load(A/'local/data/spatial.npy');part=np.full(N,-1,np.int64);raw=[];minv=np.full(N,9);maxv=np.full(N,-9);periods=['T1','T2','T3','T4']
for ti,t in enumerate(periods):
 d=dict(np.load(A/f'local/data/unet/{t}_pixels.npz'));spl=sp.ravel()[d['pixel']];np.minimum.at(minv,d['node'],spl);np.maximum.at(maxv,d['node'],spl);raw.append(d)
valid=(minv==maxv)&np.isin(minv,[0,1,2]);part[valid]=minv[valid];ids=np.flatnonzero(valid);remap=np.full(N,-1,np.int64);remap[ids]=np.arange(len(ids));a,b=g['pairs'].T;keep=valid[a]&valid[b]&(part[a]==part[b]);pairs=remap[g['pairs'][keep]];attrs=g['attrs'][keep];edge_y=g['target'][keep];edge_split=part[a[keep]];edge_split[edge_y<0]=-1
x=g['x'][ids];extra=np.load(A/'local/data/unet/extra.npz')['geometry'][ids];tr=part[ids]==0;xm=x[tr].mean(0);xs=x[tr].std(0).clip(1e-5);em=extra[tr].mean(0);es=extra[tr].std(0).clip(1e-5);frames=np.load(A/'local/data/unet/frames.npz')['frames'].astype(np.float32);sh=frames.shape[-2:];small=np.array(Image.fromarray(sp.astype(np.int32),mode='I').resize((sh[1],sh[0]),Image.Resampling.NEAREST));masks=np.stack([small==k for k in range(3)]).astype(np.float32)
np.savez_compressed(W/'data/graph.npz',x=(x-xm)/xs,extra=(extra-em)/es,pairs=pairs,attrs=attrs,edge_target=edge_y,edge_split=edge_split,phase=g['phase'][ids],centers=g['centers'][ids].astype(np.float32),part=part[ids],old_node_ids=ids,frames=frames,partition_frame_masks=masks)
np.savez(W/'data/train_only_normalization.npz',x_mean=xm,x_std=xs,geometry_mean=em,geometry_std=es)
node_hist=np.zeros((len(ids),3),np.int64);pixmeta=[]
for ti,t in enumerate(periods):
 d=raw[ti];gt=np.array(Image.open(R/(t+'_classes.png')));y=gt.ravel()[d['pixel']].astype(np.int64)-1;ok=valid[d['node']]&(y>=0)&(y<3)&(ti>0);q=dict(pixel=d['pixel'][ok],node=remap[d['node'][ok]],pf=d['pf'][ok],target=y[ok],split=part[d['node'][ok]],rid=d['rid'][ok]);np.savez_compressed(W/f'data/{t}_pixels.npz',**q)
 if ti:
  for k in range(3):np.add.at(node_hist[:,k],q['node'][(q['split']==0)&(q['target']==k)],1)
 pixmeta.append(dict(period=t,all_foreground_pixels=len(d['pixel']),eligible_typed_pixels=int(ok.sum()),split_counts={str(k):int((q['split']==k).sum())for k in [0,1,2]},class_by_split={str(k):np.bincount(q['target'][q['split']==k],minlength=3).tolist()for k in [0,1,2]}))
# Identity/root-state targets are from the released identity annotations; class-state targets use train pixels only.
rids=g['root_target'][ids];rsnodes=[];rsptr=[0];rstarget=[];rskeys=[]
for phase in range(1,4):
 for rid in np.unique(rids[(g['phase'][ids]==phase)&tr]):
  if rid<0:continue
  nn=np.flatnonzero((g['phase'][ids]==phase)&tr&(rids==rid)&(node_hist.sum(1)>0));hist=node_hist[nn].sum(0)
  if not len(nn)or hist.sum()<8:continue
  rsnodes.extend(nn);rsptr.append(len(rsnodes));rstarget.append(float(hist[1]>hist[0]+hist[2]));rskeys.append((int(rid),phase))
cons=[]
for i,(rid,phase)in enumerate(rskeys):
 if(rid,phase-1)in rskeys and rstarget[i]==0:cons.append([rskeys.index((rid,phase-1)),i])
np.savez(W/'data/root_state.npz',nodes=np.array(rsnodes,np.int64),ptr=np.array(rsptr,np.int64),target=np.array(rstarget,np.float32),consistency=np.array(cons,np.int64).reshape(-1,2))
# Retain only fully partition-contained supervised junction groups.
top=pickle.load(open(A/'joint/data/topology.pkl','rb'));pd=np.load(A/'joint/data/pairs.npz');armnodes=[];ptr=[0];aa=[];bb=[];fixed=[];target=[];psplit=[];groups=[]
for t,items in top['arms'].items():
 for hub,arms in items:
  oldnodes=np.concatenate([a['nodes']for a in arms]);u=np.unique(part[oldnodes]);src=next(q for q in top['groups']if q['period']==t and q['hub']==hub)
  if len(u)!=1 or u[0]<0 or src['target_option']<0:continue
  armids=[]
  for arm in arms:armids.append(len(ptr)-1);armnodes.extend(remap[arm['nodes']]);ptr.append(len(armnodes))
  indices={}
  for key,k in src['pair_indices'].items():
   i,j=map(int,key.split(','));indices[key]=len(aa);aa.append(armids[i]);bb.append(armids[j]);fixed.append(pd['raw'][k,96:]);target.append(pd['target'][k]);psplit.append(int(u[0]))
  groups.append(dict(period=t,hub=int(hub),split=int(u[0]),target_option=int(src['target_option']),options=src['options'],pair_indices=indices))
fixed=np.array(fixed,np.float32);psplit=np.array(psplit);fm=fixed[psplit==0].mean(0);fs=fixed[psplit==0].std(0).clip(1e-5);np.savez(W/'data/junctions.npz',nodes=np.array(armnodes,np.int64),ptr=np.array(ptr,np.int64),a=np.array(aa,np.int64),b=np.array(bb,np.int64),fixed=(fixed-fm)/fs,target=np.array(target,np.float32),split=psplit,fixed_mean=fm,fixed_std=fs);(W/'data/junction_groups.json').write_text(json.dumps(groups))
audit=dict(scope='New controlled internal module experiments on released B3 features; not an independent test',nodes_original=N,nodes_partition_pure=len(ids),nodes_by_split={str(k):int((part[ids]==k).sum())for k in [0,1,2]},edges_retained=len(pairs),pixel_data=pixmeta,GT_SHA256={t:sha(R/(t+'_classes.png'))for t in periods},source_graph_SHA256=sha(A/'local/data/unet/graph.npz'),spatial_split_SHA256=sha(A/'local/data/spatial.npy'),normalization_fit_on_train_only=True,edges_across_splits=0,nodes_shared_across_splits=0,latest_class_truth_used=True,fresh_module_initialization=True,frozen_priors_historical_B3_exposure=True,baseline_and_all_ablations_same_masks=True,junction_groups_by_split={str(k):sum(g['split']==k for g in groups)for k in [0,1,2]},external_GT_available=False)
(W/'reports/data_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
