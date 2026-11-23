"""Paired inference-only decoder interventions on each full-model checkpoint.
All three stages share the identical whole-field neural forward output. Separate from partitioned primary module results.
"""
from pathlib import Path
import numpy as np,torch,os,sys,json,hashlib,shutil,pickle,time,subprocess
from PIL import Image
from study_model import Network,PairHead,pool
W=Path(__file__).resolve().parents[1];OLD=Path('/workspace/work/V90_all_microtubes_20261118');A=OLD/'model_source/assets';CORE=W/'code/decoder_core';CORE.mkdir(exist_ok=True)
for f in(OLD/'deployment/portable_core').glob('*.py'):
 s=f.read_text()
 if f.name=='engine.py':s=s.replace('identity_threshold=.4','identity_threshold=.5')
 (CORE/f.name).write_text(s)
def one(seed):
 folder=W/'decoder_experiments'/str(seed);folder.mkdir(parents=True,exist_ok=True);ckpath=W/'experiments/full'/str(seed)/'best.pt';ck=torch.load(ckpath,weights_only=False);net=Network('full');net.load_state_dict(ck['backbone']);net.eval();pair=PairHead();pair.load_state_dict(ck['pair']);pair.eval();norm=np.load(W/'data/train_only_normalization.npz');g=dict(np.load(A/'local/data/unet/graph.npz'));extra=np.load(A/'local/data/unet/extra.npz')['geometry'];fr=np.load(A/'local/data/unet/frames.npz')['frames'].astype(np.float32);d={k:torch.from_numpy(g[k])for k in ['pairs','attrs','phase','centers']};d['centers']=d['centers'].float();d.update(x=torch.from_numpy(((g['x']-norm['x_mean'])/norm['x_std']).astype(np.float32)),extra=torch.from_numpy(((extra-norm['geometry_mean'])/norm['geometry_std']).astype(np.float32)),frames=torch.from_numpy(fr),part=torch.zeros(len(g['x']),dtype=torch.int64),partition_frame_masks=torch.ones((1,*fr.shape[-2:])),native_width=2541,native_height=1108)
 with torch.inference_mode():h,links,z=net(d)
 src=pickle.load(open(A/'joint/data/topology.pkl','rb'));pd=dict(np.load(A/'joint/data/pairs.npz'));jn=dict(np.load(W/'data/junctions.npz'));nodes=[];ptr=[0];aa=np.zeros(len(pd['raw']),np.int64);bb=aa.copy();groups={(v['period'],v['hub']):v for v in src['groups']}
 for t,items in src['arms'].items():
  for hub,arms in items:
   ids=[]
   for arm in arms:ids.append(len(ptr)-1);nodes.extend(arm['nodes']);ptr.append(len(nodes))
   for key,k in groups[(t,hub)]['pair_indices'].items():i,j=map(int,key.split(','));aa[k]=ids[i];bb[k]=ids[j]
 with torch.inference_mode():
  e=torch.nn.functional.normalize(pool(z,torch.tensor(nodes),torch.tensor(ptr)),dim=1);fixed=torch.from_numpy((pd['raw'][:,96:]-jn['fixed_mean'])/jn['fixed_std']);features=torch.cat([(e[aa]-e[bb]).abs(),e[aa]*e[bb],fixed],1);scores=pair(features).numpy()
 for path in ['joint/data','local/data/unet','local/full','branch/full','reference','analysis-scripts']: (folder/path).mkdir(parents=True,exist_ok=True)
 shutil.copytree(A/'base',folder/'base',dirs_exist_ok=True);shutil.copy2(A/'raw_graph.npz',folder/'raw_graph.npz');shutil.copy2(A/'local/data/unet/graph.npz',folder/'local/data/unet/graph.npz');shutil.copy2(A/'local/data/unet/ownership.npz',folder/'local/data/unet/ownership.npz');shutil.copy2(A/'local/data/spatial.npy',folder/'local/data/spatial.npy');pickle.dump(src,open(folder/'joint/data/topology.pkl','wb'));mu=np.r_[np.zeros(96),jn['fixed_mean']].astype(np.float32);sd=np.r_[np.ones(96),jn['fixed_std']].astype(np.float32);np.savez_compressed(folder/'joint/data/pairs.npz',raw=pd['raw'],x=features.numpy(),mean=mu,std=sd);torch.save(ck,folder/'joint/joint_best.pt');np.save(folder/'joint/joint_pair_logits.npy',scores);np.savez_compressed(folder/'joint/joint_inference.npz',h=h.numpy(),embedding=z.numpy(),connection=links.sigmoid().numpy());(folder/'joint/joint_training_results.json').write_text(json.dumps(dict(geometry_weight=2.,assignment_confidence=.9)));(folder/'reference/reference_provenance.json').write_text(json.dumps(dict(manual_truth_at_prediction=False,role='Input provenance only')));shutil.copy2('/workspace/LI/analysis-scripts/dataset_identity.py',folder/'analysis-scripts/dataset_identity.py');shutil.copytree('/workspace/LI/dataset/current',folder/'dataset/current',dirs_exist_ok=True)
 for t in ['T1','T2','T3','T4']:
  image=np.array(Image.open(A/f'base/unet/full/{t}_classes.png'))
  if t!='T1':
   p=np.load(A/f'local/data/unet/{t}_pixels.npz')
   with torch.inference_mode():
    for i in range(0,len(p['node']),32768):
     sl=slice(i,i+32768);image.ravel()[p['pixel'][sl]]=net.pixel_logits(h,torch.from_numpy(p['node'][sl]),torch.from_numpy(p['pf'][sl])).argmax(1).numpy().astype(np.uint8)+1
  for sub in ['local/full','branch/full']:Image.fromarray(image).save(folder/sub/(t+'_classes.png'))
 env=os.environ.copy();env.update(ROOT_REPRO_WORK=str(folder),ROOT_OBSERVED_PERIODS='T1,T2,T3,T4',ROOT_POSITION='0353-2-B3')
 for script in ['axis_consistency.py','junction_consistency.py']:subprocess.run([sys.executable,str(CORE/script)],env=env,check=True)
 (folder/'completed.json').write_text(json.dumps(dict(seed=seed,checkpoint_SHA256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),scope='Paired decoder-stage interventions; same unpartitioned whole-field neural context in every stage',stages=['branch/full','axis/full','junction/full'],manual_labels_read_at_prediction=False,identity_threshold=.5,junction_assignment_confidence=.9),indent=2));print('DECODER_COMPLETE',seed,flush=True)
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 for seed in [20261123,20261124,20261125]:
  for _ in range(720):
   p=W/'experiments/full'/str(seed)/'status.json'
   if p.exists()and json.loads(p.read_text()).get('stage')=='complete':break
   time.sleep(5)
  else:raise RuntimeError('Full model not ready')
  if not(W/'decoder_experiments'/str(seed)/'completed.json').exists():one(seed)
 (W/'DECODER_STUDY_COMPLETE.json').write_text(json.dumps(dict(completed=True,seeds=3),indent=2))
if __name__=='__main__':main()
