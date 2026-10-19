"""Reproduce the Q1-3-Z3 region recheck; do not infer an unsupported T1 pose."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1024*1024):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
 source=Path(__file__).resolve().parent;os.environ['LI_Q1_RECHECK_WORK']=str(a.work.resolve());os.environ['LI_Q1_RECHECK_RAW']=str(a.raw_dir.resolve());os.environ.setdefault('MPLCONFIGDIR','/tmp/LI_Q1_recheck_matplotlib')
 expected={'T1':'9046fbcb230965d6c0b7372ef394ecf73f5738cae9effd524fcf46e99380cd8a','T2':'add19453dec0d0b30c15933257c3d41084fc587ea24e2403c2feb222cb998ebb','T3':'3601e81b7b25b46c03ee6fc70cd0b8c561484e21dd30051aec83500768bb4017'}
 for period,digest in expected.items():assert sha(next(a.raw_dir.glob(period+'.*')))==digest,('Unexpected input for fixed Q1-3-Z3 recheck',period)
 # Reuse a matching installed full RoMa or the exact bundled upstream source.
 vendor=source/'vendor';fallback=source.parent/'all_series_localization_v2/vendor';vp=vendor if vendor.exists() else fallback
 os.environ['PYTHONPATH']=str(vp)+os.pathsep+os.environ.get('PYTHONPATH','')
 if Path('/workspace/.cache/torch/hub/checkpoints/roma_outdoor.pth').exists():os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
 def run(name):subprocess.run([sys.executable,str(source/name)],check=True,env=os.environ)
 # The input hashes are fixed. Cached numeric evidence is reused, never a newly claimed inference.
 run('search.py')
 import cv2,numpy as np
 from PIL import Image
 from skimage.filters import frangi
 cv2.setNumThreads(1)
 for t in [1,2,3]:
  im=Image.open(a.work/f'T{t}_search_RGB.png');arr=np.array(im);g=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY);Image.fromarray(cv2.createCLAHE(3,(12,12)).apply(g)).convert('RGB').save(a.work/f'T{t}_match_CLAHE.png')
  im.thumbnail((1288,1000));g=cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY)/255.;v=frangi(g,sigmas=[1,2,3],black_ridges=False);v=np.clip(v/max(np.quantile(v,.99),1e-6),0,1);np.save(a.work/f'T{t}_vessels.npy',v)
 run('enhanced_search.py');run('descriptor_probe.py');run('analyze.py');run('select_root_pose.py');run('build_result.py');run('infer_posed.py');run('build_result.py')
 import torch
 model_metadata=json.loads((source/'model_provenance.json').read_text())
 for weight in model_metadata['weights']:assert sha(Path(torch.hub.get_dir())/'checkpoints'/weight['file'])==weight['sha256'],('Model hash mismatch',weight['file'])
 audit=json.loads((a.work/'deliverable/evaluation/recheck_audit.json').read_text());assert audit['T1_link']['T1_to_T2_transform'] is None;assert audit['T2_to_T3_root_similarity']['anchors']>=20;assert audit['pair_QC']['conditional_on_saved_similarity']
 subprocess.run([sys.executable,str(source/'verify_result.py'),'--work',str(a.work)],check=True,env=os.environ)
 print('Rechecked Q1-3-Z3. Only T2/T3 pose supported; T1 is explicitly unresolved. No other locations written.',flush=True)

if __name__=='__main__':main()
