"""Immutable U-Net temporal anchor, source24 targets and label-free local edit candidates."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from PIL import Image
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/B3_original_UNet_latestCG_upload24_v82_20261028';CACHE=ROOT/'results/B3_UNet_junction_ConvLSTM_GAT_upload24_v76_20261028';BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028';ANCHOR=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028';PERIODS=['T1','T2','T3','T4'];SEED=2026102812
UPLOAD=Path('/workspace/attachments/c64a3bc7-6dc9-4dbd-a57b-166f682547e9/root_annotation_progress (24).json')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def refresh_prior():
 # Geometry and targets come from24; local temporal priors now describe the designated V75 output.
 f=OUT/'data/unet';g=dict(np.load(f/'graph.npz'));raw=np.load(ROOT/'results/B3_temporal_GAT_new_growth_v68_20261028/data/unet/graph.npz')['features'];N=len(g['x']);stats=np.zeros((N,7),np.float32);meanpf=np.zeros((N,15),np.float32);counts=np.zeros(N,np.float32)
 for ti,t in enumerate(PERIODS):
  p=dict(np.load(f/(t+'_pixels.npz')));n=p['node'];cnt=np.bincount(n,minlength=N);counts+=cnt
  if ti:
   q=np.load(ANCHOR/'unet/full'/(t+'_probability.npz'))['probability'].astype(np.float32);probs=q.reshape(3,-1).T[p['pixel']];probs/=probs.sum(1).clip(1e-7)[:,None]
  else:probs=np.full((len(n),3),1/3,np.float32)
  p['pf'][:,:3]=np.log(probs.clip(1e-5));np.savez_compressed(f/(t+'_pixels.npz'),**p)
  mu=np.stack([np.bincount(n,weights=probs[:,k],minlength=N)/cnt.clip(1) for k in range(3)],1);var=np.stack([np.bincount(n,weights=probs[:,k]**2,minlength=N)/cnt.clip(1) for k in range(3)],1)-mu**2;ids=g['phase']==ti;stats[ids,:3]=mu[ids];stats[ids,3:6]=np.sqrt(var[ids].clip(0));stats[ids,6]=-(mu[ids]*np.log(mu[ids].clip(1e-7))).sum(1)
  for k in range(15):meanpf[:,k]+=np.bincount(n,weights=p['pf'][:,k],minlength=N)
 x=np.c_[raw,stats];mu=x[g['node_split']==0].mean(0);sd=x[g['node_split']==0].std(0);sd[sd<1e-4]=1;g['x']=((x-mu)/sd).astype(np.float32);np.savez_compressed(f/'graph.npz',**g);e=dict(np.load(f/'extra.npz'));e['node_pf_mean']=meanpf/counts.clip(1)[:,None];np.savez_compressed(f/'extra.npz',**e)

def bridge_candidates(a,sk,gate,ti):
 out=np.zeros(a.shape,bool)
 if not ti:return out
 prev=np.array(Image.open(ANCHOR/'unet/full'/f'{PERIODS[ti-1]}_classes.png'))>0;pd=ndi.distance_transform_edt(~prev);lab,num=ndi.label(a==3,np.ones((3,3),bool));size=np.bincount(lab.ravel());arc=np.bincount(lab[sk],minlength=num+1)
 for k in np.flatnonzero((size>0)&(size<=180)&(arc<=32)&(arc>=2)):
  if k==0:continue
  yy,xx=np.where((lab==k)&sk)
  if len(xx)<2:continue
  xy=np.c_[xx,yy];vec=np.linalg.svd(xy-xy.mean(0),full_matrices=False)[2][0];proj=xy@vec;lo,hi=xy[proj.argmin()],xy[proj.argmax()];minimum=np.linalg.norm(lo-hi)
  if minimum<2:continue
  oy,ox=np.where((a==1)&ndi.binary_dilation(lab==k,iterations=3));q=np.c_[ox,oy]
  if len(q)==0:continue
  left=(np.linalg.norm(q-lo,axis=1)<=4)&((q-lo)@vec<1);right=(np.linalg.norm(q-hi,axis=1)<=4)&((q-hi)@vec>-1)
  take=lab==k
  if left.any() and right.any() and (pd[take]<=2.5).mean()>=.8:out[take]=True
 return out

def temporal_persistence_evidence(t,eligible):
 # Derived training/audit evidence; manual IDs NEVER enter prediction features.
 import json
 ti=PERIODS.index(t);out=np.zeros(eligible.shape,bool)
 if not ti:return out
 doc=json.loads(UPLOAD.read_text());st=doc['root_structure_annotation'];known=sorted(r['root_id'] for r in st['root_track_hierarchy'] if r.get('annotation_status_v60')=='user_defined');lookup={r:i for i,r in enumerate(known)};ident={};spatial=np.load(OUT/'data/spatial.npy')
 for tt in PERIODS:
  f=next(f for f in st['periods'] if f['period']==tt);r=f['owner_map_rle'];own=np.repeat(np.array(r[::2],np.int32),np.array(r[1::2],np.int64)).reshape(1108,2541);lut=np.full(int(own.max())+1,-1,np.int32)
  for seg in f['segments']:
   if seg['root_id'] in lookup:lut[seg['index']]=lookup[seg['root_id']]
  ident[tt]=lut[own];mask=np.array(Image.open(OUT/'data/reference'/f'{tt}_mask.png'))==255;ident[tt][~mask]=-1
 current=ident[t]
 for split in [0,1,2]:
  region=spatial==split;past=np.zeros(current.shape,bool);later=np.zeros(current.shape,bool)
  for j,tt in enumerate(PERIODS):
   if j==ti:continue
   valid=(ident[tt]>=0)&region;dist,nearest=ndi.distance_transform_edt(~valid,return_indices=True);matched=(dist<=2.5)&(current>=0)&(current==ident[tt][tuple(nearest)])&region
   if j<ti:past|=matched
   else:later|=matched
  # Future visibility alone cannot establish earlier persistence. T4 uses earlier evidence only.
  out|=eligible&region&past&(later if ti<3 else True)
 return out

def new_attachment_evidence(t,eligible):
 # New lateral belongs to its own current birth state at its attachment, not to the mother's class.
 ti=PERIODS.index(t);out=np.zeros(eligible.shape,bool)
 if not ti:return out
 doc=json.loads(UPLOAD.read_text());st=doc['root_structure_annotation'];known={r['root_id']:r for r in st['root_track_hierarchy'] if r.get('annotation_status_v60')=='user_defined'};relations={(r['root_id'],r.get('parent_root_id')) for r in known.values() if r.get('parentage_confirmed') and r.get('parent_root_id') in known};frame=next(f for f in st['periods'] if f['period']==t);segs={s['id']:s for s in frame['segments']};spatial=np.load(OUT/'data/spatial.npy');r=frame['owner_map_rle'];owner=np.repeat(np.array(r[::2],np.int32),np.array(r[1::2],np.int64)).reshape(1108,2541);ref=np.array(Image.open(OUT/'data/reference'/f'{t}_classes.png'));earlier={}
 for tt in PERIODS[:ti]:
  f=next(f for f in st['periods'] if f['period']==tt);r=f['owner_map_rle'];o=np.repeat(np.array(r[::2],np.int32),np.array(r[1::2],np.int64)).reshape(1108,2541);earlier[tt]=(o,{s['root_id']:[] for s in f['segments']},f['segments'])
 for node in frame['nodes']:
  arms=node.get('arms',[]);rr=[segs[a['segment_id']]['root_id'] for a in arms if a.get('segment_id') in segs]
  if len(arms)!=3:continue
  x,y=int(node['x']),int(node['y']);split=int(spatial[y,x])
  if split<0:continue
  for child,parent in relations:
   if rr.count(parent)<2 or rr.count(child)!=1:continue
   indices=[q['index'] for q in frame['segments'] if q['root_id']==child];take=np.isin(owner,indices)&(spatial==split)&np.isin(ref,[1,2,3]);n=int(take.sum())
   if n<20 or (ref[take]==2).mean()<.95:continue
   was_known=False
   for tt,(o,_,segments) in earlier.items():
    prev_indices=[q['index'] for q in segments if q['root_id']==child]
    if prev_indices and (np.isin(o,prev_indices)&(spatial==split)).sum()>=3:was_known=True;break
   if was_known:continue
   y0,y1=max(0,y-2),min(1108,y+3);x0,x1=max(0,x-2),min(2541,x+3);yy,xx=np.ogrid[y0:y1,x0:x1];out[y0:y1,x0:x1]|=((xx-x)**2+(yy-y)**2<=4)&eligible[y0:y1,x0:x1]&(spatial[y0:y1,x0:x1]==split)
 return out

def main():
 sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
 assert require_active_position('0353-2-B3')=='0353-2-B3';assert not (OUT/'initial.pt').exists(),'Existing learned run cannot be re-prepared';assert sha(UPLOAD)=='cb5a9326b73003a81b44961300de88806378a771514e886faff168a715bf2c8e'
 OUT.mkdir(exist_ok=True);(OUT/'source').mkdir(exist_ok=True);shutil.copytree(CACHE/'data',OUT/'data',dirs_exist_ok=True);(OUT/'frozen_unet').mkdir(exist_ok=True);(OUT/'candidates').mkdir(exist_ok=True)
 refresh_prior();g=dict(np.load(OUT/'data/unet/graph.npz'));route=dict(np.load(OUT/'data/unet/routing.npz'));sets={s:[] for s in ['train','val','test']};audit=[];evidence=[];immutable=[dict(path=str(UPLOAD),sha256=sha(UPLOAD)),dict(path=str(ANCHOR/'unet/best_temporal_head.pt'),sha256=sha(ANCHOR/'unet/best_temporal_head.pt'))];assert immutable[-1]['sha256']=='58a460c25b29ad4a0fdf3f5ca3e9cad3ec186af0c3827fecc942ba784955e382'
 for ti,t in enumerate(PERIODS):
  src=ANCHOR/'unet/full'/f'{t}_classes.png';a=np.array(Image.open(src));assert np.array_equal(a,np.array(Image.open(ROOT/'results/B3_ConvLSTM_GAT_upload20_v74_20261028/baseline_original/unet/full'/f'{t}_classes.png'))),'Anchor differs from ORIGINAL U-Net shown in V75 before';shutil.copy2(src,OUT/'frozen_unet'/src.name);immutable.append(dict(path=str(src),sha256=sha(src)))
  p=dict(np.load(OUT/'data/unet'/(t+'_pixels.npz')));owners=np.load(OUT/'data/unet/ownership.npz')[t];gate=owners>=0;assert np.array_equal(gate,a>0)
  sk=skeletonize(gate);degree=ndi.convolve(sk.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')-sk
  junction=sk&(degree>=3);d=ndi.distance_transform_edt(~junction);connection=(a==3)&(d<=4) if ti else np.zeros(a.shape,bool)
  # Short green/yellow islands within unbranched native paths are candidates, never a voted replacement label.
  island=np.zeros(a.shape,bool)
  if ti:
   ng=route['native_group'];counts=np.bincount(p['node'],minlength=len(g['x']));hist=np.zeros((len(g['x']),3),np.int64)
   np.add.at(hist,(p['node'],a.ravel()[p['pixel']]-1),1);chain_hist=np.zeros((ng.max()+1,3),np.int64);np.add.at(chain_hist,ng,hist)
   for c,other in [(2,3),(3,2),(3,1)]:
    # Preserve long terminal extension and all major segments. Only <=16-axis-pixel islands qualify.
    lab,num=ndi.label(a==c,np.ones((3,3),bool));sizes=np.bincount(lab.ravel());sl=np.bincount(lab[sk],minlength=num+1)
    for label in np.flatnonzero((sizes>0)&(sizes<=72)&(sl<=16)):
     if not label:continue
     yy,xx=np.where(lab==label);node=owners[yy,xx];chains=np.unique(ng[node]);same=np.isin(ng[owners.clip(0)],chains)&gate
     # Candidate evidence is outside the island, derived only from frozen U-Net colors and native geometry.
     near=ndi.binary_dilation(lab==label,iterations=3)&same
     if ((a==other)&near).sum()>=3 and sum(chain_hist[ch,other-1] for ch in chains)>=3*len(yy):island[yy,xx]=True
  persistent_bridge=bridge_candidates(a,sk,gate,ti);geom=np.load(OUT/'data/unet/extra.npz')['raw_geometry'];arms=geom[owners.clip(0),0];nearnew=ndi.distance_transform_edt(a!=2)<=8;attachment=(gate&(d<=2)&nearnew&(arms>.625)&(arms<.875)) if ti else np.zeros(gate.shape,bool);eligible=connection|island|persistent_bridge|attachment;kind=np.zeros((*a.shape,2),np.float32);kind[:,:,0]=connection|attachment;kind[:,:,1]=island|persistent_bridge
  np.savez_compressed(OUT/'candidates'/f'{t}.npz',eligible=eligible,connection=connection,island=island,persistent_bridge=persistent_bridge,new_attachment=attachment)
  ix=np.flatnonzero(eligible.ravel()[p['pixel']]);anchor=a.ravel()[p['pixel']].astype(np.int64)-1
  feat=np.c_[p['pf'],np.eye(3,dtype=np.float32)[anchor] if ti else np.zeros((len(anchor),3),np.float32),kind.reshape(-1,2)[p['pixel']]].astype(np.float32)
  weaknew=new_attachment_evidence(t,attachment);weak=temporal_persistence_evidence(t,eligible)&~weaknew;target=p['target'][ix].astype(np.int64).copy();spl=p['split'][ix].copy();wf=weak.ravel()[p['pixel'][ix]];wn=weaknew.ravel()[p['pixel'][ix]];spatial=np.load(OUT/'data/spatial.npy').ravel()[p['pixel'][ix]];refmask=np.array(Image.open(OUT/'data/reference'/f'{t}_mask.png')).ravel()[p['pixel'][ix]]==255;extra_train=(wf|wn)&((target<0)|(target>=3))&(spatial==0)&refmask;target[extra_train]=np.where(wn[extra_train],1,0);spl[extra_train]=0;local=dict(node=p['node'][ix],pf=feat[ix],target=target,anchor=anchor[ix],pixel=p['pixel'][ix],split=spl,rid=p['rid'][ix],weak_persistent=wf,weak_new_attachment=wn,original_target=p['target'][ix].astype(np.int64));np.savez_compressed(OUT/'data/unet'/(t+'_local.npz'),**local)
  for ii in np.flatnonzero(wf|wn):
   flat=int(local['pixel'][ii]);evidence.append(dict(period=t,x=flat%2541,y=flat//2541,original_annotation_class=int(local['original_target'][ii])+1,unchanged_U_Net_class=int(local['anchor'][ii])+1,suggested_class=2 if wn[ii] else 1,evidence='confirmed-parent new-lateral attachment; no earlier same-ID observation' if wn[ii] else 'same manually confirmed root axis present earlier and verified in later period when available',status='derived weak supervision/proposal; source24 unchanged'))
  for k,s in enumerate(sets):
   use=local['split']==k;sets[s].append({key:val[use] for key,val in local.items()})
  audit.append(dict(period=t,connection_candidates=int(connection.sum()),short_island_candidates=int(island.sum()),persistent_bridge_candidates=int(persistent_bridge.sum()),new_attachment_candidates=int(attachment.sum()),strong_new_attachment_candidates=int(wn.sum()),strong_history_persistent_candidates=int(wf.sum()),weak_labels_added_to_training=int(extra_train.sum()),eligible_pixels=int(eligible.sum()),root_pixels=int(gate.sum()),fraction_eligible=float(eligible.sum()/max(gate.sum(),1))))
 for s,parts in sets.items():
  arr={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};np.savez_compressed(OUT/'data/unet'/('local_'+s+'.npz'),**arr)
  print(s,'eligible typed pixels',len(arr['node']),'labels',np.bincount(arr['target'],minlength=3),'errors',int((arr['target']!=arr['anchor']).sum()),flush=True)
 if evidence:csvout(OUT/'Temporal_evidence_proposals.csv',evidence)
 for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
 js(OUT/'preparation.json',dict(source='24',source_sha256=sha(UPLOAD),learned_module_source='Latest V81 source24-trained ConvLSTM/GAT local corrector selected13',learned_module_sha256='ce29d142a03c5b4ee0bf0ca9ace14e033e5bada5e1b548c8fa24ea57769646bc',anchor='ORIGINAL U-Net in V75 BEFORE column: V66 selected37; old-reference20 newly-observed Dice .8229386284; frozen before local correction',periods=audit,scope='Only existing yellow junction pixels and short new/extension or persistent/extension anomalies; history-supported bridge candidates; no global class refinement',candidate_generation_uses_reference=False,manual_IDs_targets_only=True,geometry_frozen=True,temporal_main_classes_frozen_outside_candidates=True,source_revision=json.loads(UPLOAD.read_text()).get('annotation_revision'),immutable_inputs=immutable,outer_partition='One B3 development position, four periods; fixed inner spatial masks, no independent test'))
 print('PREPARED',OUT,audit,flush=True)
if __name__=='__main__':main()
