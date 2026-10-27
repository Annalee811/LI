"""GAT inputs use geometry/RGB/frozen model outputs; root IDs are targets only."""
from pathlib import Path
import csv,hashlib,json,shutil,sys
from collections import Counter
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
R=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;O=R/'results/B3_temporal_gat_v65_20261027';BASE=R/'results/B3_temporal_paper_v44_20261024';RGBROOT=R/'results/B3_temporal_initialization_v43_20261024/data/upper';UPLOAD=Path('/workspace/attachments/408457f2-70e1-4765-8e75-cf3f3398c685/root_annotation_progress (17).json')
PERIODS=['T1','T2','T3','T4'];MODELS=['unet','deeplab','segformer'];SEED=20261027

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as fd:
  w=csv.DictWriter(fd,fieldnames=list(rows[0]) if rows else ['empty']);w.writeheader();w.writerows(rows)
def unpack(v,shape,dtype=np.uint8):return np.repeat(np.array(v[::2],dtype),np.array(v[1::2],np.int64)).reshape(shape)
def main():
 sys.path.insert(0,str(R/'analysis-scripts'));from dataset_identity import require_active_position
 assert require_active_position('0353-2-B3')=='0353-2-B3'
 if (O/'data/preparation.json').exists():print('DATA_ALREADY_PREPARED');return
 raw=UPLOAD.read_bytes();shutil.copy2(UPLOAD,O/'data/source_progress_17.json');d=json.loads(raw);st=d['root_structure_annotation'];hs={h['root_id']:h for h in st['root_track_hierarchy']};confirmed={k:h for k,h in hs.items() if h.get('annotation_status_v60')=='user_defined'};assert len(confirmed)==142
 rgb={t:np.array(Image.open(RGBROOT/f'{t}_RGB.png').convert('RGB')) for t in PERIODS};frames={f['period']:f for f in st['periods']};H,W=1108,2541
 gt={};owners={};node_map={t:np.full((H,W),-1,np.int32) for t in PERIODS};nodepoints=[];nodes=[];feat=[];native_adj=[];pixel_indices=[];hist=[];inputs=[UPLOAD]
 lut=np.array([255,2,3,1,4],np.uint8)
 for ti,t in enumerate(PERIODS):
  im=next(im for im in d['images'] if im['name']==t+'_RGB.png');assert sha(RGBROOT/(t+'_RGB.png'))==im['image_sha256'];inputs.append(RGBROOT/(t+'_RGB.png'))
  root=unpack(im['layers']['mask'],(H,W))>0;tmp=unpack(im['layers']['temporal'],(H,W));g=np.zeros((H,W),np.uint8);g[root]=lut[tmp[root]];gt[t]=g;owners[t]=unpack(frames[t]['owner_map_rle'],(H,W),np.int32)
  assert np.array_equal(owners[t]>0,root)
  # Sort original ownership indices once. Each root pixel stays on its own
  # original native segment; nearest chunk only subdivides computation.
  flat=owners[t].ravel();fg=np.flatnonzero(flat);order=np.argsort(flat[fg],kind='stable');fg=fg[order];vals=flat[fg];starts=np.r_[0,np.flatnonzero(vals[1:]!=vals[:-1])+1];ends=np.r_[starts[1:],len(vals)];byidx={int(vals[a]):fg[a:b] for a,b in zip(starts,ends)}
  for s in frames[t]['segments']:
   pp=np.array(s['points'],np.int32);boundaries=list(range(0,len(pp),24));ids=[];local_points=[];local_chunk=[]
   for a in boundaries:
    q=pp[a:a+24];nid=len(nodes);ids.append(nid);nodepoints.append(q);local_points.extend(q.tolist());local_chunk.extend([nid]*len(q));h=confirmed.get(s['root_id']);nodes.append(dict(node=nid,period=t,period_index=ti,segment_id=s['id'],native_segment_index=s['index'],chunk_start=a,chunk_end=min(a+24,len(pp)),root_target=s['root_id'] if h else None,manual_label=h['label'] if h else None,family_target=h['family_root_id'] if h else None,parent_target=h.get('parent_root_id') if h else None,parentage_confirmed=bool(h and h.get('parentage_confirmed')),points=q.tolist()))
   native_adj.extend(zip(ids[:-1],ids[1:]));owned=byidx.get(s['index'],np.array([],np.int64));yy,xx=np.divmod(owned,W);nearest=cKDTree(np.array(local_points)).query(np.c_[xx,yy])[1] if len(owned) else np.array([],int);nn=np.array(local_chunk)[nearest];node_map[t].ravel()[owned]=nn
   for nid in ids:
    q=nodepoints[nid].astype(float);sel=owned[nn==nid];pixel_indices.append(sel);counts=np.bincount(g.ravel()[sel],minlength=256)[1:4];hist.append(counts)
    center=q.mean(0);v=q[-1]-q[0];l=np.linalg.norm(v);v=v/max(l,1);arc=np.linalg.norm(np.diff(q,axis=0),axis=1).sum();dir2=(q[len(q)//2]-q[0]);dir2/=max(1,np.linalg.norm(dir2));curv=abs(float(v[0]*dir2[1]-v[1]*dir2[0]));area=len(sel)
    pix=rgb[t].reshape(-1,3)[sel].astype(float)/255 if len(sel) else rgb[t][q[:,1].astype(int),q[:,0].astype(int)].astype(float)/255
    prev=rgb[PERIODS[max(0,ti-1)]].reshape(-1,3)[sel].astype(float)/255 if len(sel) else pix
    features=[center[0]/W,center[1]/H,q[0,0]/W,q[0,1]/H,q[-1,0]/W,q[-1,1]/H,arc/40,l/40,float(v[0]),float(v[1]),curv,min(area/max(arc,1),20)/20,np.log1p(area)/8,*pix.mean(0),*pix.std(0),*(pix-prev).mean(0),*np.abs(pix-prev).mean(0),*np.eye(4)[ti]]
    feat.append(features)
  assert np.all(node_map[t].ravel()[fg]>=0)
 np.savez_compressed(O/'data/geometry.npz',features=np.array(feat,np.float32),hist=np.array(hist,np.float32),**{'node_map_'+t:node_map[t] for t in PERIODS},**{'gt_'+t:gt[t] for t in PERIODS})
 centers=np.array([np.array(q).mean(0) for q in nodepoints]);unit=np.array([(q[-1]-q[0])/max(1,np.linalg.norm(q[-1]-q[0])) for q in nodepoints]);period=np.array([n['period_index'] for n in nodes]);pairs={tuple(sorted((int(a),int(b)))) for a,b in native_adj}
 for ti in range(4):
  ids=np.flatnonzero(period==ti);tree=cKDTree(centers[ids]);dd,qq=tree.query(centers[ids],k=min(9,len(ids)))
  for a,dist,others in zip(ids,dd,qq):
   for dis,b in zip(np.atleast_1d(dist),np.atleast_1d(others)):
    if dis<=65 and a!=ids[b]:pairs.add(tuple(sorted((int(a),int(ids[b])))))
  if ti:
   prev=np.flatnonzero(period==ti-1);tree=cKDTree(centers[prev]);dd,qq=tree.query(centers[ids],k=min(8,len(prev)))
   for a,dist,others in zip(ids,dd,qq):
    for dis,b in zip(np.atleast_1d(dist),np.atleast_1d(others)):
     if dis<=90:pairs.add(tuple(sorted((int(a),int(prev[b])))))
 pair=np.array(sorted(pairs),np.int64);attrs=[];targets=[];same_native=[]
 native_set={tuple(sorted((int(a),int(b)))) for a,b in native_adj}
 for a,b in pair:
  p,q=nodepoints[a],nodepoints[b];mind=float(cKDTree(p).query(q)[0].min());delta=centers[b]-centers[a];td=abs(int(period[a]-period[b]));na,nb=nodes[a],nodes[b];attrs.append([min(mind/40,3),min(np.linalg.norm(delta)/100,3),abs(float(unit[a]@unit[b])),delta[0]/100,delta[1]/100,td,float(na['segment_id']==nb['segment_id'] and td==0),abs(len(p)-len(q))/24])
  ta,tb=na['root_target'],nb['root_target'];target=-1
  if ta and tb:
   target=1 if ta==tb else (2 if ((na['parent_target']==tb and na['parentage_confirmed']) or (nb['parent_target']==ta and nb['parentage_confirmed'])) else 0)
  targets.append(target);same_native.append((a,b) in native_set)
 attrs=np.array(attrs,np.float32);targets=np.array(targets,np.int64);same_native=np.array(same_native,bool)
 # Keep every date and every side of one manually defined family together.
 families=sorted({h['family_root_id'] for h in confirmed.values()});best=None
 for attempt in range(300):
  rng=np.random.default_rng(SEED+attempt);order=rng.permutation(families);n=len(order);fm={f:(0 if i<int(n*.6) else 1 if i<int(n*.8) else 2) for i,f in enumerate(order)};split=np.array([fm.get(n['family_target'],-1) for n in nodes],np.int8);hc=np.array([np.array(hist)[split==k].sum(0) for k in range(3)])
  if hc.min()<50:continue
  sizes=np.array([sum(fm[h['family_root_id']]==k for h in confirmed.values()) for k in range(3)]);score=float(np.log1p(hc).sum()-abs(sizes/142-np.array([.6,.2,.2])).sum()*12)
  if best is None or score>best[0]:best=(score,fm,split,hc,sizes,attempt)
 assert best is not None,'Cannot obtain family-separated partitions containing all three classes'
 _,fm,split,hc,sizes,attempt=best;edge_split=np.full(len(pair),-1,np.int8)
 for k in range(3):edge_split[(split[pair[:,0]]==k)&(split[pair[:,1]]==k)&(targets>=0)&(~same_native)]=k
 print('NODES',len(nodes),'PAIRS',len(pair),'FAMILIES',len(families),'ROOT SPLITS',sizes.tolist(),'CLASS PIXELS',hc.tolist(),flush=True)
 assert all(np.any((edge_split==k)&(targets==1)) and np.any((edge_split==k)&(targets==0)) for k in range(3))
 np.savez_compressed(O/'data/graph.npz',pairs=pair,edge_attr=attrs,pair_target=targets,edge_split=edge_split,node_split=split,native_adjacency=same_native)
 js(O/'data/nodes.json',nodes);js(O/'data/family_split.json',dict(seed=SEED,attempt=int(attempt),families={f:['train','val','test'][k] for f,k in fm.items()},root_counts=sizes.tolist(),class_pixel_counts=hc.tolist(),all_periods_and_same_family_same_split=True,transductive_features_only=True))
 human_rows=[]
 for rid,h in confirmed.items():
  for t in PERIODS:
   ss=[s for s in frames[t]['segments'] if s['root_id']==rid];human_rows.append(dict(human_label=h['label'],root_id=rid,parent_label=hs.get(h.get('parent_root_id'),{}).get('label',''),family_label=hs[h['family_root_id']]['label'],period=t,observed=bool(ss),native_length_px=round(sum(s['length_px'] for s in ss),3),original_segments=len(ss),split=['train','val','test'][fm[h['family_root_id']]],annotation_status=h['annotation_status_v60']))
 csvout(O/'Human_root_IDs_four_periods.csv',human_rows)
 rng=np.random.default_rng(SEED);samples=[];sample_nodes=[];sample_y=[];sample_period=[]
 for nid,(meta,indices) in enumerate(zip(nodes,pixel_indices)):
  if meta['period']=='T1' or meta['root_target'] is None:continue
  y0=gt[meta['period']].ravel()[indices];indices=indices[(y0>=1)&(y0<=3)];
  if not len(indices):continue
  chosen=indices if len(indices)<=16 else rng.choice(indices,16,replace=False);samples.extend(chosen.tolist());sample_nodes.extend([nid]*len(chosen));sample_y.extend((gt[meta['period']].ravel()[chosen]-1).tolist());sample_period.extend([meta['period_index']]*len(chosen))
 np.savez_compressed(O/'data/samples.npz',pixel=np.array(samples),node=np.array(sample_nodes),target=np.array(sample_y),period=np.array(sample_period))
 for m in MODELS:
  mfeat=[];baselogs=[];samplelog=np.empty((len(samples),3),np.float32);gatecounts=[]
  for ti,t in enumerate(PERIODS):
   clpath=BASE/m/'full'/f'{t}_classes.png';inputs.append(clpath);cl=np.array(Image.open(clpath));p=np.full((3,H,W),1/3,np.float32)
   if ti:
    path=BASE/m/'full'/f'{t}_probability.npz';inputs.append(path);p=np.load(path)['probability'].astype(np.float32);p/=np.maximum(p.sum(0),1e-7)
   flatp=p.reshape(3,-1).T;selected=np.array(sample_period)==ti
   if selected.any():samplelog[selected]=np.log(np.maximum(flatp[np.array(samples)[selected]],1e-5))
   gate=cl.ravel()>0
   for nid in np.flatnonzero(period==ti):
    inds=pixel_indices[nid];q=flatp[inds] if len(inds) else np.array([[1/3]*3]);mu=q.mean(0);sd=q.std(0);entropy=-(q*np.log(np.maximum(q,1e-5))).sum(1);support=gate[inds].mean() if len(inds) else 0
    mfeat.append(np.r_[mu,sd,np.quantile(q,[.1,.9],axis=0).ravel(),entropy.mean(),support]);baselogs.append(np.log(np.maximum(mu,1e-5)));gatecounts.append(support)
  # Frames/nodes were appended in chronological order.
  x=np.c_[np.array(feat,np.float32),np.array(mfeat,np.float32)];mu=x[split==0].mean(0);sd=x[split==0].std(0);sd[sd<1e-4]=1;x=(x-mu)/sd
  np.savez_compressed(O/f'data/{m}_features.npz',x=x.astype(np.float32),baseline_log=np.array(baselogs,np.float32),sample_log=samplelog,normalization_mean=mu,normalization_std=sd)
  print('FEATURES',m,x.shape,'supervised sampled pixels',len(samples),flush=True)
 inputs=list(dict.fromkeys(inputs));immutable=[dict(path=str(p),bytes=p.stat().st_size,SHA256=sha(p)) for p in inputs]
 feature_names=['cx','cy','start_x','start_y','end_x','end_y','arc_length','chord_length','tangent_x','tangent_y','curvature','width_proxy','area_log']+[f'rgb_mean_{k}' for k in range(3)]+[f'rgb_std_{k}' for k in range(3)]+[f'rgb_delta_{k}' for k in range(3)]+[f'rgb_abs_delta_{k}' for k in range(3)]+[f'period_{k}' for k in range(4)]+[f'base_probability_mean_{k}' for k in range(3)]+[f'base_probability_std_{k}' for k in range(3)]+[f'base_probability_quantile_{k}' for k in range(6)]+['base_entropy','base_support']
 assert len(feature_names)==x.shape[1] and not any('root_id' in n or 'label' in n or 'parent' in n for n in feature_names)
 prep=dict(position='0353-2-B3',source_upload='root_annotation_progress (17).json',source_SHA256=hashlib.sha256(raw).hexdigest(),source_model_assistance=d.get('annotation_revision'),confirmed_human_root_IDs=142,assistant_candidates_excluded=12,unassigned_candidates_excluded=True,graph_nodes=len(nodes),graph_pairs=len(pair),feature_names=feature_names,input_root_IDs_or_parent_IDs_or_label_colors=False,edges_from_geometry_only=True,human_connections_not_used_as_input=True,chunk_points=24,graph_is_transductive_features_only=True,all_periods_of_each_manual_family_same_partition=True,family_split_seed=SEED+int(attempt),node_histogram_by_split=hc.tolist(),root_counts_by_split=sizes.tolist(),pair_targets={str(k):Counter(targets[edge_split==k].tolist()) for k in range(3)},temporal_classes={'1':'persistent','2':'newly_observed','3':'tip_extension'},backbone_and_saved_head_frozen=True,base_version='V44/V43 saved 100-epoch heads',training_epochs=100,training_seed=SEED,immutable_inputs=immutable,evaluation_scope='Single B3 field; geometry-conditional association and transductive GAT refinement, not external generalization; model-assisted pixel reference retained')
 js(O/'data/preparation.json',prep);js(O/'stage1_complete.json',dict(state='data_prepared',**{k:prep[k] for k in ['source_SHA256','confirmed_human_root_IDs','graph_nodes','graph_pairs','root_counts_by_split']}))
 for f in HERE.glob('*.py'):shutil.copy2(f,O/'source'/f.name)
 print('DATA_PREPARED',flush=True)
if __name__=='__main__':main()
