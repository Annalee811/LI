"""Publish isolated experiment outputs; never replace the manual editor or old results."""
import base64,hashlib,json,subprocess,shutil
from pathlib import Path
from prepare import O,R,MODELS,js,sha
HERE=Path(__file__).resolve().parent;REPO='repos/Annalee811/LI';BRANCH='annotation-restore-v50-delivery';PREFIX='roma-results/0353-2/0353-2-B3/temporal-gat-upload17-v65'

def api(path,obj=None,method=None):
 command=['gh','api',REPO+('/'+path if path else '')]
 if obj is not None:command+=['--input','-']
 if method:command+=['--method',method]
 r=subprocess.run(command,cwd=R,input=None if obj is None else json.dumps(obj),capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr[:1600])
 return json.loads(r.stdout)
def main():
 v=json.loads((O/'final_verification.json').read_text());assert v['passed'];viewer=json.loads((O/'viewer_verification.json').read_text());assert viewer['passed'] and viewer['tested_viewer_SHA256']==sha(O/'Root_ID_comparison.html');assert not api('')['private']
 for p in HERE.glob('*'):
  if p.is_file():shutil.copy2(p,O/'source'/p.name)
 names=['README.md','RESULTS.md','Root_ID_comparison.html','Human_root_IDs_four_periods.csv','Root_ID_comparison_all_models.csv','Root_ID_tracking_metrics.csv','Temporal_class_metrics_before_after.csv','Temporal_macro_metrics_before_after.csv','metrics_summary.json','learning_verification.json','final_verification.json','viewer_verification.json','evaluation_protocol.json','ID_figure_ROI.json','stage1_complete.json','training_complete.json','evaluation_complete.json','figure_manifest.json','ablation_figure_manifest.json','class_figure_panels.json']
 names += [p.name for p in sorted(O.glob('*.tif'))]
 names += [str(p.relative_to(O)) for p in sorted((O/'data').iterdir()) if p.is_file() and p.suffix in ['.npz','.json']]
 names += [str(p.relative_to(O)) for p in sorted((O/'source').iterdir()) if p.is_file() and p.suffix in ['.py','.mjs']]
 for m in MODELS:
  for variant in ['archived','mlp','gat']:
   directory=O/(m+'_'+variant)
   for name in ['initial.pt','best.pt','resume.pt','config.json','completion.json','history.csv','inference.npz','node_track.npy','track_registry.json','Root_ID_comparison.csv','test_identity_metrics.json','test_tracking_metrics.json','test_temporal_metrics.json','test_ROC.json','test_confusion.npy']:
    if (directory/name).exists():names.append(m+'_'+variant+'/'+name)
   if variant!='archived':
    names += [str(p.relative_to(O)) for p in sorted((directory/'full').glob('*_classes.png'))]
    if (O/(m+'_'+variant+'.log')).exists():names.append(m+'_'+variant+'.log')
 names=list(dict.fromkeys(names));assert all('viewer_test' not in n and 'browser-profile' not in n for n in names)
 cachepath=O/'published_blob_state.json';cache=json.loads(cachepath.read_text()) if cachepath.exists() else {};entries=[];records=[]
 for i,name in enumerate(names):
  data=(O/name).read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
  if cache.get(blob)!=len(data):
   assert api('git/blobs',dict(encoding='base64',content=base64.b64encode(data).decode()))['sha']==blob
   assert base64.b64decode(api('git/blobs/'+blob)['content'])==data;cache[blob]=len(data);js(cachepath,cache)
  path=PREFIX+'/'+name;entries.append(dict(path=path,mode='100644',type='blob',sha=blob));records.append(dict(path=path,bytes=len(data),SHA256=hashlib.sha256(data).hexdigest(),blob=blob));print('VERIFIED',i+1,'/',len(names),name,len(data),flush=True)
 head=api('git/ref/heads/'+BRANCH)['object']['sha'];mh=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest();statepath=O/'publication_state.json';state=json.loads(statepath.read_text()) if statepath.exists() else {}
 if state.get('commit')==head and state.get('manifest_SHA256')==mh:commit=head
 else:
  tree=api('git/trees',dict(base_tree=api('git/commits/'+head)['tree']['sha'],tree=entries))['sha'];commit=api('git/commits',dict(message='Train actual100-epoch GAT and no-GAT controls on uploaded17 root identities; preserve before/after class-specific results and failed improvements',tree=tree,parents=[head]))['sha'];js(statepath,dict(commit=commit,parent=head,manifest_SHA256=mh));api('git/refs/heads/'+BRANCH,dict(sha=commit,force=False),method='PATCH')
 assert api('git/ref/heads/'+BRANCH)['object']['sha']==commit;tree=api('git/trees/'+commit+'?recursive=1');assert not tree.get('truncated');actual={v['path']:v['sha'] for v in tree['tree'] if v['type']=='blob'};assert all(actual[r['path']]==r['blob'] for r in records)
 base='https://raw.githubusercontent.com/Annalee811/LI/'+commit+'/'+PREFIX+'/';links=dict(commit=commit,verification='Public GitHub API byte read-back and committed path verification',archive_url='https://github.com/Annalee811/LI/tree/'+commit+'/'+PREFIX,viewer_url=base+'Root_ID_comparison.html',human_IDs_url=base+'Human_root_IDs_four_periods.csv',ID_comparison_url=base+'Root_ID_comparison_all_models.csv',ID_metrics_url=base+'Root_ID_tracking_metrics.csv',class_metrics_url=base+'Temporal_class_metrics_before_after.csv',new_roots_url=base+'Fig_New_roots_before_after_GAT.tif',persistent_roots_url=base+'Fig_Persistent_roots_before_after_GAT.tif',growing_tips_url=base+'Fig_Growing_tips_before_after_GAT.tif',ID_figure_url=base+'Fig_Root_ID_comparison.tif',ablation_url=base+'Fig_GAT_MLP_temporal_ablation.tif',ID_ablation_url=base+'Fig_Root_ID_tracking_ablation.tif',table_url=base+'Table_Three_classes_before_after_GAT.tif',monitoring_url=base+'Fig_GAT_training_monitoring.tif',report_url=base+'RESULTS.md',files=records)
 js(O/'download_links.json',links);print(json.dumps({k:v for k,v in links.items() if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
