"""Meaningful safeguards: numeric metrics, GAT equivariance, label-input audit."""
from pathlib import Path
import copy,importlib.util,json,tempfile
import numpy as np
import torch
from model import Model
from prepare import O,R,js,sha
from metrics import binary_curve,adjusted_rand

def main():
 checks=[]
 def check(name,value):
  checks.append(dict(name=name,passed=bool(value)));assert value,name;print('PASS',name,flush=True)
 check('tie-aware AUC/AP formulas match known cases',binary_curve([0,1],[.5,.5])['AUC']==.5 and binary_curve([0,1],[.5,.5])['average_precision']==.5 and binary_curve([0,1],[.1,.9])['AUC']==1)
 torch.set_num_threads(2);torch.manual_seed(17);model=Model(43,'gat').eval();n=8;x=torch.randn(n,43);pairs=torch.tensor([[0,1],[1,2],[2,4],[4,6],[6,7]]);edge=torch.cat([pairs,pairs.flip(1),torch.stack([torch.arange(n),torch.arange(n)],1)],0);attrs=torch.randn(len(edge),8);pa=torch.randn(len(pairs),8)
 with torch.inference_mode():a=model(x,edge.T,attrs,pairs,pa)
 perm=torch.randperm(n);inv=torch.empty_like(perm);inv[perm]=torch.arange(n)
 with torch.inference_mode():b=model(x[perm],inv[edge].T,attrs,inv[pairs],pa)
 check('GAT follows node permutation rather than numeric node identity',torch.allclose(a[0],b[0][inv],atol=2e-6) and torch.allclose(a[1],b[1],atol=2e-6) and torch.allclose(a[2],b[2][inv],atol=2e-6))
 check('GAT forward outputs are finite',all(torch.isfinite(z).all() for z in a))
 # Rebuild the feature graph with every root ID monotonically renamed and all
 # human labels replaced. This is a software audit artifact, never training data.
 source=json.loads((O/'data/source_progress_17.json').read_text());changed=copy.deepcopy(source);st=changed['root_structure_annotation'];ids=sorted(h['root_id'] for h in st['root_track_hierarchy']);rename={rid:'R'+str(500000+i) for i,rid in enumerate(ids)}
 for i,h in enumerate(st['root_track_hierarchy']):
  h['root_id']=rename[h['root_id']];h['label']='AUDIT-'+str(i).zfill(4);h['family_root_id']=rename[h['family_root_id']];h['parent_root_id']=rename.get(h.get('parent_root_id'))
 for f in st['periods']:
  for s in f['segments']:s['root_id']=rename[s['root_id']]
 tmp=Path(tempfile.mkdtemp(prefix='gat65_label_input_audit_'));(tmp/'data').mkdir();(tmp/'source').mkdir();upload=tmp/'software_test_annotation.json';upload.write_text(json.dumps(changed));path=Path(__file__).with_name('prepare.py');spec=importlib.util.spec_from_file_location('gat65_audit_prepare',path);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p);p.O=tmp;p.UPLOAD=upload;p.main()
 original=np.load(O/'data/graph.npz');audit=np.load(tmp/'data/graph.npz');check('renaming root IDs and human numbers leaves geometric edges and supervised partition unchanged',all(np.array_equal(original[k],audit[k]) for k in ['pairs','edge_attr','pair_target','edge_split','node_split','native_adjacency']))
 for m in ['unet','deeplab','segformer']:
  a=np.load(O/f'data/{m}_features.npz');b=np.load(tmp/f'data/{m}_features.npz');check(m+' neural inputs are byte-value-identical after all root IDs/numbers are replaced',all(np.array_equal(a[k],b[k]) for k in a.files))
 check('uploaded17 original file and model-assisted provenance remain preserved',sha(O/'data/source_progress_17.json')==json.loads((O/'data/preparation.json').read_text())['source_SHA256'])
 js(O/'learning_verification.json',dict(passed=True,checks=checks,temporary_renamed_data_not_training_or_deliverable=True,audit_directory=str(tmp)));print('LEARNING_VERIFIED',flush=True)
if __name__=='__main__':main()
