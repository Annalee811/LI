"""Endpoint continuation learning, using cached ConvLSTM/GAT representations."""
import csv,hashlib,json,pickle,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from topology import build,matchings
ROOT=Path('/workspace/LI');HERE=Path(__file__).resolve().parent;OUT=ROOT/'results/B3_root_axis_junction_v83_20261028';LAST=ROOT/'results/B3_original_UNet_latestCG_upload24_v82_20261028';BASE=ROOT/'results/B3_temporal_upload18_noGAT_v66_20261028';PERIODS=['T1','T2','T3','T4'];SEED=2026102813

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rows):
 if not rows:return
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 sys.path.insert(0,str(ROOT/'analysis-scripts'));from dataset_identity import require_active_position
 assert require_active_position('0353-2-B3')=='0353-2-B3';OUT.mkdir(exist_ok=True);(OUT/'data').mkdir(exist_ok=True);(OUT/'source').mkdir(exist_ok=True)
 g=dict(np.load(LAST/'data/unet/graph.npz'));cached=dict(np.load(LAST/'inference.npz'));owners=dict(np.load(LAST/'data/unet/ownership.npz'));h=cached['h'];emb=cached['embedding'];spatial=np.load(LAST/'data/spatial.npy');raw=np.load(ROOT/'results/B3_temporal_GAT_new_growth_v68_20261028/data/unet/graph.npz')['features'];rgb={t:np.array(Image.open(BASE/'data/upper'/f'{t}_RGB.png')).astype(np.float32)/255 for t in PERIODS};topos={};trees={};tangents={};mapping={}
 for ti,t in enumerate(PERIODS):
  tp=build(owners[t]>=0);topos[t]=tp;pts=[];vec=[];segid=[]
  for si,s in enumerate(tp['segments']):
   p=s['points'];left=p[np.maximum(np.arange(len(p))-5,0)];right=p[np.minimum(np.arange(len(p))+5,len(p)-1)];v=right-left;v=v/np.linalg.norm(v,axis=1).clip(1e-8)[:,None];pts.extend(p);vec.extend(v);segid.extend([si]*len(p))
  pts=np.array(pts);tangents[t]=np.array(vec);trees[t]=cKDTree(pts);mapping[t]=np.array(segid)
  print('TOPOLOGY',t,'segments',len(tp['segments']),'hubs',len(tp['hubs']),flush=True)
 def support(t,pts,direction):
  dist,nn=trees[t].query(pts,k=4);align=np.abs((tangents[t][nn]*direction).sum(-1));return float(((dist<=3.5)&(align>=.9)).any(1).mean())
 rows=[];features=[];targets=[];splits=[];groups=[];armrecords={}
 for ti,t in enumerate(PERIODS):
  tp=topos[t];armrecords[t]=[]
  for ji,hub in enumerate(tp['hubs']):
   arms=hub['arms'];n=len(arms)
   if n not in [3,4] or any(len(a['body_points'])<4 for a in arms):continue
   x,y=np.rint(hub['center']).astype(int);split=int(spatial[y,x]);aa=[]
   for a in arms:
    p=a['body_points'];node=np.unique(owners[t][p[:,1],p[:,0]]);node=node[node>=0];e=emb[node].mean(0);e/=max(np.linalg.norm(e),1e-8);width=float(raw[node,6].mean());color=rgb[t][p[:,1],p[:,0]].mean(0);past=[support(tt,p,a['direction']) for tt in PERIODS];ids=g['root_target'][node[g['node_split'][node]==split]];known=ids[ids>=0];rid=-1
    if len(known) and len(known)/max(len(ids),1)>=.6:
     values,counts=np.unique(known,return_counts=True)
     if counts.max()/len(known)>=.9:rid=int(values[counts.argmax()])
    aa.append(dict(embedding=e,width=width,color=color,support=np.array(past),rid=rid,nodes=node,**a))
   start=len(features);pairs=[];local={}
   for i in range(n):
    for j in range(i+1,n):
     a,b=aa[i],aa[j];opposition=float(-a['direction']@b['direction']);gap=np.linalg.norm(a['point']-b['point'])/16;f=np.r_[np.abs(a['embedding']-b['embedding']),a['embedding']*b['embedding'],opposition,gap,abs(a['width']-b['width']),np.abs(a['color']-b['color']),a['support']+b['support'],np.abs(a['support']-b['support']),np.eye(4)[ti]].astype(np.float32);features.append(f);valid=split>=0 and a['rid']>=0 and b['rid']>=0;targets.append(int(a['rid']==b['rid']) if valid else -1);splits.append(split if valid else -1);local[(i,j)]=len(features)-1;pairs.append((i,j));rows.append(dict(period=t,hub=ji,arm_a=i,arm_b=j,opposition=opposition,split=split,target=targets[-1]))
   opts=matchings(n);truth=[]
   for option in opts:
    good=all(aa[i]['rid']>=0 and aa[j]['rid']>=0 and aa[i]['rid']==aa[j]['rid'] for i,j in option)
    involved={i for pair in option for i in pair};unused=set(range(n))-involved
    if n==3:good &= all(aa[next(iter(unused))]['rid']>=0 and aa[next(iter(unused))]['rid']!=aa[i]['rid'] for i in involved)
    truth.append(bool(good))
   groups.append(dict(period=t,hub=ji,split=split,pair_indices={str(i)+','+str(j):k for (i,j),k in local.items()},options=opts,target_option=truth.index(True) if sum(truth)==1 else -1))
   armrecords[t].append((ji,aa))
 x=np.array(features);target=np.array(targets);split=np.array(splits);mu=x[split==0].mean(0);sd=x[split==0].std(0);sd[sd<1e-5]=1;np.savez_compressed(OUT/'data/pairs.npz',x=((x-mu)/sd).astype(np.float32),raw=x,target=target,split=split,mean=mu,std=sd);csvout(OUT/'data/pair_targets.csv',rows)
 with (OUT/'data/topology.pkl').open('wb') as f:pickle.dump(dict(topos=topos,groups=groups,arms=armrecords),f)
 js(OUT/'preparation.json',dict(source24_sha256=sha(LAST/'data/source_progress_24.json'),baseline='Original U-Net plus previously delivered local V82 result; frozen binary geometry',module='Frozen latest ConvLSTM/GAT embeddings; learn a junction-continuation relation head; structured segment decoder',feature_dimension=x.shape[1],pair_targets={s:np.bincount(target[(split==k)&(target>=0)],minlength=2).tolist() for k,s in enumerate(['train','val','test'])},unique_matching_junctions={s:sum(q['split']==k and q['target_option']>=0 for q in groups) for k,s in enumerate(['train','val','test'])},manual_IDs_input=False,user_case_ROI=[1024,704,1280,960],case_scope='User-reported regression case; no independent test claim'))
 import shutil
 for p in HERE.glob('*.py'):shutil.copy2(p,OUT/'source'/p.name)
 print('PAIRS',x.shape,'targets',np.bincount(target[target>=0],minlength=2),flush=True)
if __name__=='__main__':main()
