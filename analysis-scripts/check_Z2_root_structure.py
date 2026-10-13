#!/usr/bin/env python3
"""Last structural-view diagnostic for Z2's unmatched T1/T2; never changes original RGB."""
from pathlib import Path
import sys,os,json,gc,shutil
os.environ.setdefault('TORCH_HOME','/workspace/.cache/torch')
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,cv2,torch
from PIL import Image
from types import SimpleNamespace
import register_roma as reg
from run_0353_1_root_priority import support,postprocess,read
ROOT=Path.cwd()/'results/microtube_registration';POSITION='0353-1-Z2';RAW=ROOT/'input/raw/V-变化'/POSITION
cv2.setNumThreads(2);torch.set_num_threads(2);torch.manual_seed(20261013)
class StructuralModel:
 def __init__(self,m):self.m=m
 def normalize(self,value):
  rgb=np.array(value.convert('RGB')) if isinstance(value,Image.Image) else read(value)
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);gray=cv2.createCLAHE(2,(12,12)).apply(gray)
  gray=cv2.GaussianBlur(gray,(3,3),.55)
  ridge=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21))).astype(np.float32)
  candidates,_=support(rgb);root=cv2.resize(candidates.astype(np.uint8),tuple(reversed(gray.shape)),interpolation=cv2.INTER_NEAREST)>0
  factor=np.where(root,1.,.35);image=np.clip(18+ridge*factor*4.+gray*.15,0,255).astype(np.uint8)
  return Image.fromarray(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB))
 def match(self,a,b,**kw):return self.m.match(self.normalize(a),self.normalize(b),**kw)
 def sample(self,*a,**kw):return self.m.sample(*a,**kw)
from romatch import roma_outdoor
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.set_float32_matmul_precision('highest')
base=roma_outdoor(device=device,coarse_res=560,upsample_res=864,symmetric=True,use_custom_corr=False,upsample_preds=True)
paths=[next(f for f in RAW.iterdir() if f.stem==f'T{i}') for i in [1,2]]
folder=ROOT/'0353-1_root_priority_refinements'/POSITION/'root_structure_T1_to_T2'
args=SimpleNamespace(prealignments={},device=device,confidence=.5,cycle=3.,samples=5000,seed=20261013,ransac=3.,coarse=560,upsample=864,tile=128)
b,c,v,m=reg.pair(StructuralModel(base),paths[0],paths[1],folder,args);del b,c,v;gc.collect()
m['inference_only_normalization']='Grayscale CLAHE and bright-ridge response; root candidate support weight1, soil .35, inference only. Output sampled from original RGB.'
postprocess(folder,ROOT/'0353-1_root_priority_configuration'/POSITION,0,m)
baseline=ROOT/POSITION/'T1_to_T2';old=json.loads((baseline/'metrics.json').read_text());new=json.loads((folder/'metrics.json').read_text())
if new['effective_area_ratio']>=.01 and new['reliable_sampled_matches']>=100 and new['effective_area_ratio']>old['effective_area_ratio']:
 for f in folder.iterdir():
  if f.is_file():shutil.copy2(f,baseline/f.name)
 mf=ROOT/POSITION/'manifest.json';d=json.loads(mf.read_text());d['structural_first_pair_retry']=dict(selected=True,effective_area_ratio=new['effective_area_ratio'],inference_only_normalization=m['inference_only_normalization']);mf.write_text(json.dumps(d,ensure_ascii=False,indent=2));print('SELECTED_STRUCTURAL_VIEW',new['effective_area_ratio'],flush=True)
else:print('NO_RELIABLE_STRUCTURAL_ALIGNMENT',new['effective_area_ratio'],new['reliable_sampled_matches'],flush=True)
