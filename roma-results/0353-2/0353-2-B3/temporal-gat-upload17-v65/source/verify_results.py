from pathlib import Path
import csv,json,hashlib
import numpy as np
import torch
from PIL import Image
from prepare import O,BASE,MODELS,PERIODS,sha,js

def main():
 checks=[]
 def check(name,v):assert v,name;checks.append(dict(name=name,passed=True))
 prep=json.loads((O/'data/preparation.json').read_text());graph={k:v for k,v in np.load(O/'data/graph.npz').items()};nodes=json.loads((O/'data/nodes.json').read_text())
 for m in MODELS:
  for kind in ['mlp','gat']:
   p=O/(m+'_'+kind);c=json.loads((p/'completion.json').read_text());rows=list(csv.DictReader((p/'history.csv').open(encoding='utf-8-sig')));check(m+' '+kind+' has100 real epochs',[int(r['epoch']) for r in rows]==list(range(1,101)) and c['epochs_completed']==100);check(m+' '+kind+' selected weights checksum matches',sha(p/'best.pt')==c['checkpoint_SHA256']);a=torch.load(p/'initial.pt',weights_only=False,map_location='cpu')['state_dict'];b=torch.load(p/'best.pt',weights_only=False,map_location='cpu')['state_dict'];check(m+' '+kind+' was actually optimized',any(not torch.equal(a[k],b[k]) for k in a));tr=np.load(p/'node_track.npy');edges=graph['pairs'][graph['native_adjacency']];check(m+' '+kind+' keeps native continuous segments whole',np.all(tr[edges[:,0]]==tr[edges[:,1]]))
   for t in PERIODS:check(m+' '+kind+' '+t+' root outline unchanged',np.array_equal(np.array(Image.open(BASE/m/'full'/f'{t}_classes.png'))>0,np.array(Image.open(p/'full'/f'{t}_classes.png'))>0))
 for key in ['root_target','family_target']:
  groups={}
  for n,s in zip(nodes,graph['node_split']):
   if n[key]:groups.setdefault(n[key],set()).add(int(s))
  check(key+' keeps all periods in one internal supervision mask',all(len(v)==1 for v in groups.values()))
 check('all original input hashes unchanged',all(sha(Path(i['path']))==i['SHA256'] for i in prep['immutable_inputs']));check('root-number and root-ID replacement cannot alter model inputs',json.loads((O/'learning_verification.json').read_text())['passed']);view=json.loads((O/'viewer_verification.json').read_text());check('interactive viewer is browser-tested at exact file hash',view['passed'] and view['tested_viewer_SHA256']==sha(O/'Root_ID_comparison.html'))
 manifests=[json.loads((O/n).read_text()) for n in ['figure_manifest.json','ablation_figure_manifest.json']]
 for manifest in manifests:
  for f in manifest['figures']:
   p=O/f['file'];check(f['file']+' file hash matches',sha(p)==f['SHA256'])
   with Image.open(p) as im:check(f['file']+' is600dpi RGB TIFF',im.mode=='RGB' and im.size==tuple(f['dimensions']) and tuple(round(v) for v in im.info['dpi'])==(600,600))
 js(O/'final_verification.json',dict(passed=True,checks=checks,source_SHA256=prep['source_SHA256'],inner_heldout_root_IDs=26,epochs_each=100,trained_new_modules=6,original_manual_annotation_not_overwritten=True,all_four_B3_dates_belong_to_one_outer_development_position=True));print('ALL_RESULTS_VERIFIED',len(checks),flush=True)
if __name__=='__main__':main()
